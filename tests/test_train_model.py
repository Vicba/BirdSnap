import pytest
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train_model.model_builder import get_model

def test_get_model_returns_module():
    """Test if get_model returns a PyTorch model."""
    num_classes = 10
    model = get_model(num_classes)
    assert isinstance(model, torch.nn.Module), "The returned model is not a PyTorch Module." # nosec

def test_get_model_output_features():
    """Test if the output features of the final fully connected layer match num_classes."""
    num_classes = 10
    model = get_model(num_classes)
    assert isinstance(model.fc, torch.nn.Linear), "The final layer is not a Linear layer." # nosec
    assert model.fc.out_features == num_classes, f"Expected {num_classes} output features but got {model.fc.out_features}." # nosec
