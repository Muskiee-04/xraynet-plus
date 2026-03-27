import torch
import torch.onnx
from torchvision import models
import torch.nn as nn
import os

def convert_to_onnx():
    """Convert trained PyTorch model to ONNX (NIH ChestX-ray14 multi-label head = 15 classes)."""
    
    # Check if trained model exists
    if not os.path.exists('models/saved/best_model.pth'):
        print("❌ No trained model found. Please train the model first.")
        return
    
    # Load the trained model
    checkpoint = torch.load('models/saved/best_model.pth', map_location='cpu')
    
    # Create model architecture (NIH multi-label)
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 15)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = "models/saved/xraynet_plus.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model successfully converted to ONNX: {onnx_path}")
    print(f"📁 File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_to_onnx()