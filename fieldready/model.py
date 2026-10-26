import random
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PREPROCESS = {'resize': 256, 'crop': 224, 'mean': [0.485, 0.456, 0.406],
              'std': [0.229, 0.224, 0.225], 'interpolation': 'bilinear', 'color': 'RGB'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_model(pretrained=False, num_classes=3):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def image_transform(training=False, settings=None):
    p = settings or PREPROCESS
    spatial = [transforms.RandomResizedCrop(p['crop'], scale=(0.75, 1.0)),
               transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
               transforms.ColorJitter(0.15, 0.15, 0.15, 0.03)] if training else [
                   transforms.Resize(p['resize'], interpolation=InterpolationMode.BILINEAR),
                   transforms.CenterCrop(p['crop'])]
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(p['mean'], p['std'])])
