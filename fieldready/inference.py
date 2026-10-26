"""Reusable inference; never downloads pretrained weights."""
import json
from pathlib import Path
import torch
from PIL import ImageOps
from .model import build_model, image_transform
from .data import CLASSES, OTHER_CLASS

UNCERTAIN = 'Uncertain — take a clearer photo or consult an expert'


class Predictor:
    def __init__(self, model_dir='artifacts/fieldready', device='cpu'):
        directory = Path(model_dir)
        if not (directory / 'metadata.json').is_file() or not (directory / 'best.pt').is_file():
            raise FileNotFoundError(f'Missing trained model in {directory}. Run python -m fieldready.train first.')
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata['architecture'] != 'mobilenet_v3_small' or self.metadata['classes'] not in [CLASSES, CLASSES + [OTHER_CLASS]]:
            raise ValueError('Unsupported model architecture or class order')
        self.classes = self.metadata['classes']
        self.device = torch.device(device)
        self.model = build_model(num_classes=len(self.classes)).to(self.device)
        checkpoint = torch.load(directory / 'best.pt', map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.transform = image_transform(settings=self.metadata['preprocessing'])
        self.threshold = float(self.metadata['threshold'])

    @torch.inference_mode()
    def predict(self, image):
        if image is None:
            return {'message': 'Upload a tomato leaf photo.', 'scores': {}, 'accepted': False}
        image = ImageOps.exif_transpose(image).convert('RGB')
        scores = self.model(self.transform(image).unsqueeze(0).to(self.device)).softmax(1)[0].cpu().tolist()
        index = max(range(len(scores)), key=scores.__getitem__)
        accepted = index < 3 and scores[index] >= self.threshold
        return {'message': self.classes[index] if accepted else UNCERTAIN,
                'scores': dict(zip(self.classes, scores)), 'accepted': accepted}
