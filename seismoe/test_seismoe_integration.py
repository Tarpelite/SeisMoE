#!/usr/bin/env python3
"""
Test script for SeisMoE integration
"""

import sys
import os
sys.path.append('benchmark')

def test_model_import():
    """Test if SeisMoELit can be imported"""
    try:
        from models import SeisMoELit
        print("✓ SeisMoELit imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import SeisMoELit: {e}")
        return False

def test_config_loading():
    """Test if config files are valid"""
    import json
    
    configs_to_test = [
        'configs/stead_seismoe.json',
        'configs/ethz_seismoe.json'
    ]
    
    for config_file in configs_to_test:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ {config_file} loaded successfully")
            print(f"  Model: {config['model']}")
            print(f"  Data: {config['data']}")
            print(f"  Experts: {config['model_args']['num_experts']}")
        except Exception as e:
            print(f"✗ Failed to load {config_file}: {e}")
            return False
    
    return True

def test_model_instantiation():
    """Test if SeisMoELit can be instantiated"""
    try:
        from models import SeisMoELit
        
        # Test with minimal parameters
        model = SeisMoELit(
            base_model="original",  # Use original instead of stead to avoid download
            num_experts=2,
            num_experts_per_token=1,
            lr=1e-3
        )
        print("✓ SeisMoELit instantiated successfully")
        print(f"  Number of experts: {model.model.num_experts}")
        print(f"  Experts per token: {model.model.num_experts_per_token}")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate SeisMoELit: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("SeisMoE Integration Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_model_import),
        ("Config Loading Test", test_config_loading),
        ("Model Instantiation Test", test_model_instantiation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"Test failed: {test_name}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed! SeisMoE is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)