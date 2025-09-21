# SeisMoE: EQTransformer with Mixture of Experts

## 概述

SeisMoE是EQTransformer的增强版本，采用了Mixture of Experts (MoE)架构来提高模型的表达能力和专业化程度。

## 主要特性

### 1. MoE架构
- **多专家系统**: 将标准的FFN层替换为多个专家网络
- **智能路由**: 使用门控机制动态选择专家
- **负载均衡**: 确保专家使用的均匀分布

### 2. 改进点

**相比原始实现的改进:**
- ✅ 优化了专家路由逻辑，提高计算效率
- ✅ 改进了负载均衡损失函数
- ✅ 集成到PyTorch Lightning框架
- ✅ 支持参数冻结（只训练MoE层）
- ✅ 详细的损失监控和日志记录

### 3. 配置参数

```json
{
  "model": "SeisMoE",
  "model_args": {
    "base_model": "stead",           // 基础预训练模型
    "num_experts": 4,                // 专家数量
    "num_experts_per_token": 2,      // 每个token使用的专家数
    "moe_loss_coef": 0.01,          // MoE负载均衡损失系数
    "lr": 1e-3,                     // 学习率
    "freeze_non_moe": false         // 是否冻结非MoE参数
  }
}
```

## 使用方法

### 1. 训练SeisMoE模型

```bash
# 使用STEAD数据集训练
python benchmark/train.py configs/stead_seismoe.json

# 使用ETHZ数据集微调（冻结非MoE参数）
python benchmark/train.py configs/ethz_seismoe.json
```

### 2. 评估模型

```bash
# 评估SeisMoE模型
python benchmark/eval.py --model_path weights/SeisMoE_STEAD/model.ckpt --config configs/stead_seismoe.json
```

### 3. 在代码中使用

```python
from benchmark.models import SeisMoELit

# 创建模型
model = SeisMoELit(
    base_model="stead",
    num_experts=8,
    num_experts_per_token=2,
    moe_loss_coef=0.02,
    freeze_non_moe=True  # 只训练MoE层
)

# 训练步骤会自动处理MoE损失
losses = model.shared_step(batch)
print(f"Total loss: {losses['loss']}")
print(f"MoE loss: {losses['moe_loss']}")
```

## 配置文件说明

### 基础配置 (stead_seismoe.json)
```json
{
  "model": "SeisMoE",
  "data": "STEAD",
  "trainer_args": {
    "accelerator": "gpu",
    "devices": 4,
    "max_epochs": 100
  },
  "model_args": {
    "base_model": "stead",
    "num_experts": 4,
    "num_experts_per_token": 2,
    "moe_loss_coef": 0.01,
    "lr": 1e-3,
    "rotate_array": true,
    "freeze_non_moe": false
  }
}
```

### 微调配置 (ethz_seismoe.json)
```json
{
  "model": "SeisMoE",
  "data": "ETHZ", 
  "trainer_args": {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 50
  },
  "model_args": {
    "base_model": "stead",
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_loss_coef": 0.02,
    "lr": 5e-4,
    "rotate_array": true,
    "freeze_non_moe": true  // 只训练MoE层
  }
}
```

## 监控和日志

训练过程中会记录以下指标：
- `train_loss` / `val_loss`: 总损失
- `train_loss_det` / `val_loss_det`: 检测损失
- `train_loss_p` / `val_loss_p`: P波损失
- `train_loss_s` / `val_loss_s`: S波损失  
- `train_moe_loss` / `val_moe_loss`: MoE负载均衡损失

## 测试

运行集成测试：
```bash
python test_seismoe_integration.py
```

## 技术细节

### MoE层实现
- **专家网络**: 每个专家是一个标准的FFN (Linear → ReLU → Dropout → Linear)
- **门控机制**: 使用线性层计算门控分数，Top-K选择专家
- **负载均衡**: 通过监控专家使用频率，计算均匀分布损失

### 初始化策略
- 从原始EQTransformer的FFN层初始化专家网络
- 添加小量噪声以确保专家多样性
- 保留原始的注意力和归一化层

### 内存优化
- 高效的专家路由算法
- 避免不必要的计算和内存分配
- 支持梯度检查点（如果需要）

## 注意事项

1. **内存使用**: MoE会增加模型大小，建议使用足够的GPU内存
2. **训练稳定性**: 负载均衡损失系数需要合理调节
3. **专家数量**: 建议从较少专家开始，逐步增加
4. **微调策略**: 可以先训练全部参数，再冻结非MoE层进行微调

## 实验建议

1. **基线对比**: 先用相同设置训练原始EQTransformer作为基线
2. **专家数量**: 尝试2, 4, 8, 16个专家
3. **负载均衡**: 调节moe_loss_coef从0.001到0.1
4. **训练策略**: 对比全量训练vs冻结训练的效果