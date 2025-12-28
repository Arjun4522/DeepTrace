#!/usr/bin/env python3
"""
Inspect a PyTorch checkpoint to understand its structure.
This helps debug model loading issues.
"""

import torch
import sys
import argparse
from pprint import pprint

class ModelConfig:
    """Configuration class for model (compatibility with saved checkpoints)"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def inspect_checkpoint(checkpoint_path):
    """Inspect a PyTorch checkpoint file"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    try:
        # Load with weights_only=False to handle custom classes
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n1. CHECKPOINT TYPE:")
        print(f"   {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("\n2. CHECKPOINT KEYS:")
            for key in checkpoint.keys():
                print(f"   - {key}")
            
            print("\n3. KEY DETAILS:")
            
            # Check for config
            if 'config' in checkpoint:
                print("\n   CONFIG:")
                config = checkpoint['config']
                print(f"   Type: {type(config)}")
                if hasattr(config, '__dict__'):
                    print("   Attributes:")
                    for attr, value in vars(config).items():
                        print(f"     {attr}: {value}")
                else:
                    print(f"   Value: {config}")
            
            # Check for model state dict
            if 'model_state_dict' in checkpoint:
                print("\n   MODEL_STATE_DICT:")
                state_dict = checkpoint['model_state_dict']
                print(f"   Number of parameters: {len(state_dict)}")
                print("   First 10 parameter names:")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    shape = state_dict[key].shape
                    print(f"     {i+1}. {key}: {shape}")
            
            # Check for model directly
            if 'model' in checkpoint:
                print("\n   MODEL:")
                model_data = checkpoint['model']
                print(f"   Type: {type(model_data)}")
                if isinstance(model_data, dict):
                    print(f"   Number of parameters: {len(model_data)}")
                    print("   First 10 parameter names:")
                    for i, key in enumerate(list(model_data.keys())[:10]):
                        shape = model_data[key].shape
                        print(f"     {i+1}. {key}: {shape}")
            
            # Check for other keys
            if 'epoch' in checkpoint:
                print(f"\n   EPOCH: {checkpoint['epoch']}")
            
            if 'optimizer_state_dict' in checkpoint:
                print(f"\n   OPTIMIZER_STATE_DICT: Present")
            
            if 'loss' in checkpoint:
                print(f"\n   LOSS: {checkpoint['loss']}")
            
            # Check for any other interesting keys
            other_keys = [k for k in checkpoint.keys() 
                         if k not in ['config', 'model_state_dict', 'model', 
                                     'epoch', 'optimizer_state_dict', 'loss']]
            if other_keys:
                print(f"\n   OTHER KEYS: {other_keys}")
        
        else:
            # Checkpoint is the state dict itself
            print("\n2. STATE DICT (direct):")
            print(f"   Number of parameters: {len(checkpoint)}")
            print("   First 10 parameter names:")
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                shape = checkpoint[key].shape
                print(f"     {i+1}. {key}: {shape}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS:")
        print("=" * 70)
        
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                print("✓ Checkpoint has config - ModelInference will use it")
            else:
                print("⚠ No config found - will use default architecture")
            
            if 'model_state_dict' in checkpoint:
                print("✓ Found 'model_state_dict' key")
            elif 'model' in checkpoint:
                print("✓ Found 'model' key")
            else:
                print("⚠ No standard model key found")
        else:
            print("⚠ Checkpoint is raw state dict (no config)")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoint structure"
    )
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint file (.pth)"
    )
    
    args = parser.parse_args()
    
    success = inspect_checkpoint(args.checkpoint)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()