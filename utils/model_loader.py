"""
Model loading utilities.
"""

import json
import torch
import torch.nn as nn
import timm
from configs.app_config import MODEL_CONFIG


class AudioClassifier(nn.Module):
    """
    EfficientNet-based audio classifier for multi-label classification.
    """
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        
        # Load EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimension
        self.feature_dim = 1280  # EfficientNet-B0 feature dim
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def load_model(device='cpu'):
    """
    Load trained model.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    # Load metadata to get number of classes
    with open(MODEL_CONFIG['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    n_classes = metadata['n_classes']
    
    # Create model
    model = AudioClassifier(n_classes=n_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(
        MODEL_CONFIG['model_path'],
        map_location=device,
        weights_only=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_metadata():
    """
    Load model metadata.
    
    Returns:
        Dictionary with metadata
    """
    with open(MODEL_CONFIG['metadata_path'], 'r') as f:
        metadata = json.load(f)
    return metadata


def load_thresholds():
    """
    Load optimized classification thresholds.
    
    Returns:
        Dictionary mapping instrument names to thresholds
    """
    with open(MODEL_CONFIG['thresholds_path'], 'r') as f:
        thresholds = json.load(f)
    return thresholds


def get_device():
    """
    Get the best available device.
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
