import torch
from torchvision import models

def get_model(num_classes: int) -> torch.nn.Module:
    """Returns a resnet18 PyTorch model based on the model type and number of classes.

    Args:
      model_type: A string representing the model type.
      num_classes: An integer representing the number of classes.

    Returns:
        A PyTorch model instance.
        """
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model