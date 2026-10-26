import io
import zipfile
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from .model import seed_worker


class LeafDataset(Dataset):
    def __init__(self, rows, transform, archive=None):
        self.rows, self.transform, self.archive = rows, transform, archive
        self._zip = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.archive:
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.archive)
            source = io.BytesIO(self._zip.read(row['path']))
        else:
            source = row['path']
        # Fail loudly on corrupt images; never silently change evaluation denominators.
        with Image.open(source) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return tensor, row['label']


def loader(rows, transform, archive=None, batch_size=64, workers=2, shuffle=False, seed=42):
    return DataLoader(LeafDataset(rows, transform, archive), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def predict_loader(model, batches, device):
    model.eval()
    probabilities, targets = [], []
    for images, labels in batches:
        probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)
