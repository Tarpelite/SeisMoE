import warnings
import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models

import seisbench.models as sbm
from pathlib import Path
import yaml
import pandas as pd
import models
from util import load_best_model

# TODO: gate network tracing weights
# TODO: Add EMD feature
class MoEFeedForward(nn.Module):
    """
    Mixture of Experts FeedForward layer
    Replaces the standard FeedForward layer with multiple experts and a gating mechanism
    """
    
    def __init__(
        self, 
        io_size: int, 
        hidden_size: int = 128,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        drop_rate: float = 0.1,
        noise_std: float = 0.1,
        load_balancing_loss_coef: float = 0.01,
        init_from_ffn: Optional[nn.Module] = None
    ):
        """
        Args:
            io_size: Input/output size
            hidden_size: Hidden layer size
            num_experts: Total number of experts
            num_experts_per_token: Number of experts to use per token
            drop_rate: Dropout rate
            noise_std: Standard deviation for noise in gating
            load_balancing_loss_coef: Coefficient for load balancing loss
            init_from_ffn: Original FFN module to initialize from
        """
        super().__init__()
        
        self.io_size = io_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.noise_std = noise_std
        self.load_balancing_loss_coef = load_balancing_loss_coef
        
        # Create experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(io_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_size, io_size)
            )
            
            # Initialize from original FFN if provided
            # if init_from_ffn is not None:
            #     with torch.no_grad():
            #         # Copy weights from original FFN with some noise for diversity
            #         expert[0].weight.copy_(init_from_ffn.lin1.weight + 
            #                               torch.randn_like(init_from_ffn.lin1.weight) * 0.01)
            #         expert[0].bias.copy_(init_from_ffn.lin1.bias + 
            #                             torch.randn_like(init_from_ffn.lin1.bias) * 0.01)
            #         expert[3].weight.copy_(init_from_ffn.lin2.weight + 
            #                               torch.randn_like(init_from_ffn.lin2.weight) * 0.01)
            #         expert[3].bias.copy_(init_from_ffn.lin2.bias + 
            #                             torch.randn_like(init_from_ffn.lin2.bias) * 0.01)
            
            if init_from_ffn is not None:
                # print(f"Initializing expert {i} from original FFN")
                with torch.no_grad():
                    # 将第一个expert初始化为原始FFN的权重
                    if i == 0:
                        expert[0].weight.copy_(init_from_ffn.lin1.weight)
                        expert[0].bias.copy_(init_from_ffn.lin1.bias)
                        expert[3].weight.copy_(init_from_ffn.lin2.weight)
                        expert[3].bias.copy_(init_from_ffn.lin2.bias)
                    else:
                        # 其他expert在原始权重基础上添加小扰动
                        expert[0].weight.copy_(init_from_ffn.lin1.weight + 0.01 * torch.randn_like(init_from_ffn.lin1.weight))
                        expert[0].bias.copy_(init_from_ffn.lin1.bias + 0.01 * torch.randn_like(init_from_ffn.lin1.bias))
                        expert[3].weight.copy_(init_from_ffn.lin2.weight + 0.01 * torch.randn_like(init_from_ffn.lin2.weight))
                        expert[3].bias.copy_(init_from_ffn.lin2.bias + 0.01 * torch.randn_like(init_from_ffn.lin2.bias))

            self.experts.append(expert)
        
        # Gating network
        self.gate = nn.Linear(io_size, num_experts)
        
        # For tracking load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
        
    def forward(self, x):
        """
        x shape: (batch, channel, time)
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape for processing
        x = x.permute(0, 2, 1)  # (batch, time, channel)
        x_flat = x.reshape(-1, self.io_size)  # (batch * time, channel)
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)  # (batch * time, num_experts)
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k gating
        topk_gate_scores, topk_indices = torch.topk(
            gate_logits, self.num_experts_per_token, dim=-1
        )
        topk_gate_scores = F.softmax(topk_gate_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # More efficient expert routing
        # Create routing mask for all experts at once
        routing_weights = torch.zeros(x_flat.shape[0], self.num_experts, device=x_flat.device)
        
        # Set routing weights based on top-k selection
        for i in range(self.num_experts_per_token):
            expert_indices = topk_indices[:, i]  # (batch * time,)
            expert_scores = topk_gate_scores[:, i]  # (batch * time,)
            
            # Use scatter to efficiently set routing weights
            routing_weights.scatter_(1, expert_indices.unsqueeze(1), expert_scores.unsqueeze(1))
        
        # Process all experts
        for expert_id in range(self.num_experts):
            # Get mask for this expert
            expert_mask = routing_weights[:, expert_id] > 0
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_id](expert_input)
                expert_weights = routing_weights[expert_mask, expert_id:expert_id+1]
                
                # Weighted combination
                output[expert_mask] += expert_output * expert_weights
                
                # Track usage for load balancing
                if self.training:
                    self.expert_usage[expert_id] += expert_mask.sum().float()
                    
        if self.training:
            self.total_tokens += x_flat.shape[0]
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, channels)
        output = output.permute(0, 2, 1)  # (batch, channel, time)
        
        return output
    
    def get_load_balancing_loss(self):
        """
        Compute load balancing loss to encourage equal expert usage
        Uses coefficient of variation to measure load imbalance
        """
        if self.total_tokens > 0:
            # Normalize usage by total tokens
            expert_usage_normalized = self.expert_usage / self.total_tokens
            
            # Compute load balancing loss - encourage uniform distribution
            target_usage = 1.0 / self.num_experts
            
            # Use squared difference from uniform distribution
            load_balance_loss = torch.sum((expert_usage_normalized - target_usage) ** 2)
            
            # Alternative: coefficient of variation approach
            # mean_usage = expert_usage_normalized.mean()
            # std_usage = expert_usage_normalized.std()
            # cv_loss = std_usage / (mean_usage + 1e-8)
            
            return load_balance_loss * self.load_balancing_loss_coef
        return torch.tensor(0.0, device=self.expert_usage.device)
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.expert_usage.zero_()
        self.total_tokens.zero_()


class TransformerMoE(nn.Module):
    """
    Transformer block with MoE FeedForward layer
    """
    
    def __init__(
        self, 
        input_size: int,
        drop_rate: float = 0.1,
        attention_width: Optional[int] = None,
        eps: float = 1e-5,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        init_from_transformer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        # Copy attention and normalization layers from original
        if init_from_transformer is not None:
            self.attention = copy.deepcopy(init_from_transformer.attention)
            self.norm1 = copy.deepcopy(init_from_transformer.norm1)
            self.norm2 = copy.deepcopy(init_from_transformer.norm2)
            
            # Create MoE FFN initialized from original FFN
            self.ff = MoEFeedForward(
                io_size=input_size,
                hidden_size=128,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                drop_rate=drop_rate,
                init_from_ffn=init_from_transformer.ff
            )
        else:
            # Create new layers
            from seisbench.models.eqtransformer import SeqSelfAttention, LayerNormalization
            self.attention = SeqSelfAttention(input_size, attention_width=attention_width, eps=eps)
            self.norm1 = LayerNormalization(input_size)
            self.ff = MoEFeedForward(
                io_size=input_size,
                hidden_size=128,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                drop_rate=drop_rate
            )
            self.norm2 = LayerNormalization(input_size)
    
    def forward(self, x):
        y, weight = self.attention(x)
        y = x + y
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y2)
        return y2, weight


class SeisMoE(nn.Module):
    """
    SeisMoE: EQTransformer with Mixture of Experts
    Loads a pretrained EQTransformer and converts FFN layers to MoE
    """
    
    def __init__(
        self,
        base_model: Optional[str] = "original",
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        device: str = "cpu"
    ):
        """
        Args:
            base_model: Name or path of pretrained EQTransformer model
            num_experts: Number of experts in MoE layers
            num_experts_per_token: Number of experts to use per token
            device: Device to load model on
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.device = device
        
        # Load pretrained EQTransformer
        print(f"Loading pretrained EQTransformer: {base_model}")
        if not "weights" in base_model or base_model.endswith(".ckpt"):
            self.base_model = sbm.EQTransformer.from_pretrained(base_model).to(device)
        else:
            weights = Path(base_model)
            version = sorted(weights.iterdir())[-1]
            config_path = version / "hparams.yaml"
            with open(config_path, "r") as f:
                # config = yaml.safe_load(f)
                config = yaml.full_load(f)
            model_cls = models.__getattribute__(config["model"] + "Lit")
            print(f"Model class: {model_cls}")
            self.base_model = load_best_model(model_cls, weights, version.name).model.to(device)
            print(f"Loaded local pretrained model from {version}")
        
        # Store original transformers for initialization
        original_transformer_d0 = self.base_model.transformer_d0
        original_transformer_d = self.base_model.transformer_d
        
        # Replace Transformer layers with MoE versions
        print(f"Converting FFN layers to MoE with {num_experts} experts...")
        
        self.base_model.transformer_d0 = TransformerMoE(
            input_size=16,
            drop_rate=self.base_model.drop_rate,
            eps=1e-7 if self.base_model.original_compatible else 1e-5,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            init_from_transformer=original_transformer_d0
        ).to(device)
        
        # self.base_model.transformer_d = TransformerMoE(
        #     input_size=16,
        #     drop_rate=self.base_model.drop_rate,
        #     eps=1e-7 if self.base_model.original_compatible else 1e-5,
        #     num_experts=num_experts,
        #     num_experts_per_token=num_experts_per_token,
        #     init_from_transformer=original_transformer_d
        # ).to(device)
        
        self.base_model.transformer_d = original_transformer_d  # Keep second transformer as is for simplicity
        
        print("SeisMoE model initialized successfully!")
    
    def forward(self, x, logits=False):
        """Forward pass through the model"""
        return self.base_model(x)
    
    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        self.base_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        self.base_model.eval()
        return self
    
    def get_moe_loss(self):
        """Get load balancing loss from all MoE layers"""
        loss = 0.0
        if hasattr(self.base_model.transformer_d0.ff, 'get_load_balancing_loss'):
            loss += self.base_model.transformer_d0.ff.get_load_balancing_loss()
        if hasattr(self.base_model.transformer_d.ff, 'get_load_balancing_loss'):
            loss += self.base_model.transformer_d.ff.get_load_balancing_loss()
        return loss
    
    def reset_moe_stats(self):
        """Reset MoE usage statistics"""
        if hasattr(self.base_model.transformer_d0.ff, 'reset_usage_stats'):
            self.base_model.transformer_d0.ff.reset_usage_stats()
        if hasattr(self.base_model.transformer_d.ff, 'reset_usage_stats'):
            self.base_model.transformer_d.ff.reset_usage_stats()
    
    def annotate(self, *args, **kwargs):
        """Pass through to base model's annotate method"""
        return self.base_model.annotate(*args, **kwargs)
    
    def classify(self, *args, **kwargs):
        """Pass through to base model's classify method"""
        return self.base_model.classify(*args, **kwargs)
    
    def __getattr__(self, name):
        """Pass through attribute access to base_model for compatibility"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If attribute not found in SeisMoE, try base_model
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
            raise
    
    def save_model(self, path: str):
        """Save the MoE model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_experts': self.num_experts,
            'num_experts_per_token': self.num_experts_per_token,
            'base_model_args': self.base_model.get_model_args() if hasattr(self.base_model, 'get_model_args') else None,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved MoE model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

def test_seismoe():
    """
    Test function to verify SeisMoE model
    Compares outputs with original EQTransformer
    """
    print("=" * 60)
    print("Testing SeisMoE Model")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load original EQTransformer
    print("\n1. Loading original EQTransformer...")
    original_model = sbm.EQTransformer.from_pretrained("stead").to(device)
    original_model.eval()
    
    # Create SeisMoE from the same pretrained model
    print("\n2. Creating SeisMoE from pretrained EQTransformer...")
    moe_model = SeisMoE(
        base_model="stead",
        num_experts=4,
        num_experts_per_token=2,
        device=device
    )
    moe_model.eval()
    
    # Test input
    print("\n3. Creating test input...")
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 6000).to(device)
    print(f"   Input shape: {test_input.shape}")
    
    # Forward pass through both models
    print("\n4. Forward pass through models...")
    with torch.no_grad():
        original_outputs = original_model(test_input)
        moe_outputs = moe_model(test_input)
    
    # Compare output shapes
    print("\n5. Comparing output shapes:")
    assert len(original_outputs) == len(moe_outputs), "Number of outputs mismatch!"
    
    for i, (orig, moe) in enumerate(zip(original_outputs, moe_outputs)):
        label = ["Detection", "P-phase", "S-phase"][i]
        print(f"   {label}:")
        print(f"      Original shape: {orig.shape}")
        print(f"      MoE shape:      {moe.shape}")
        assert orig.shape == moe.shape, f"Shape mismatch for {label}!"
    
    print("\n✓ All output shapes match!")
    
    # Check value ranges (should be between 0 and 1 for probabilities)
    print("\n6. Checking output value ranges:")
    for i, (orig, moe) in enumerate(zip(original_outputs, moe_outputs)):
        label = ["Detection", "P-phase", "S-phase"][i]
        print(f"   {label}:")
        print(f"      Original range: [{orig.min().item():.4f}, {orig.max().item():.4f}]")
        print(f"      MoE range:      [{moe.min().item():.4f}, {moe.max().item():.4f}]")
        assert 0 <= moe.min() <= 1 and 0 <= moe.max() <= 1, f"Invalid probability range for {label}!"
    
    print("\n✓ All outputs in valid probability range!")
    
    # Count parameters
    print("\n7. Model parameters:")
    original_params = sum(p.numel() for p in original_model.parameters())
    moe_params = sum(p.numel() for p in moe_model.parameters())
    print(f"   Original model: {original_params:,} parameters")
    print(f"   MoE model:      {moe_params:,} parameters")
    print(f"   Increase:       {(moe_params - original_params):,} parameters "
          f"({100 * (moe_params - original_params) / original_params:.1f}%)")
    
    # Test MoE-specific features
    print("\n8. Testing MoE-specific features:")
    
    # Test load balancing loss
    moe_loss = moe_model.get_moe_loss()
    print(f"   MoE load balancing loss: {moe_loss:.6f}")
    
    # Test reset stats
    moe_model.reset_moe_stats()
    print("   ✓ MoE statistics reset successfully")
    
    # Test with Lightning module if available
    # if HAS_LIGHTNING:
    #     print("\n9. Testing PyTorch Lightning module:")
    #     lit_model = SeisMoELit(
    #         num_experts=4,
    #         num_experts_per_token=2,
    #         freeze_non_moe=True
    #     )
        
    #     # Create dummy batch
    #     dummy_batch = {
    #         "X": test_input,
    #         "y": torch.rand(batch_size, 2, 6000).to(device),
    #         "detections": torch.rand(batch_size, 1, 6000).to(device)
    #     }
        
    #     # Test forward pass
    #     losses = lit_model.shared_step(dummy_batch)
    #     print(f"   Training loss: {losses['loss'].item():.4f}")
    #     print(f"   Detection loss: {losses['loss_det'].item():.4f}")
    #     print(f"   P-phase loss: {losses['loss_p'].item():.4f}")
    #     print(f"   S-phase loss: {losses['loss_s'].item():.4f}")
    #     print(f"   MoE loss: {losses['moe_loss']:.6f}")
    #     print("   ✓ Lightning module working correctly!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    
    return moe_model

if __name__ == "__main__":
    test_seismoe()