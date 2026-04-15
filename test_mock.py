import torch
from models.dit_classifier import DiTClassifier
from config import Config

def test_dit():
    # Mock input features
    batch_size = 4
    x = torch.randn(batch_size, Config.FEATURE_DIM)
    
    # Init DiT
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=256,
        depth=2,
        num_heads=4
    )
    
    # Forward pass
    out = model(x)
    assert out.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {out.shape}"
    print("DiT Classifier shape test passed!")

if __name__ == "__main__":
    test_dit()
