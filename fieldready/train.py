"""Train on official PlantVillage training data; select on grouped validation only."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from .data import prepare, CLASSES, OTHER_CLASS, PV_REVISION
from .model import build_model, image_transform, seed_everything, PREPROCESS
from .runtime import loader, predict_loader
from .metrics import choose_threshold, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/plantvillage')
    parser.add_argument('--output', default='artifacts/fieldready')
    parser.add_argument('--include-other', action='store_true', help='Learn a fourth rejection class from other tomato diseases')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-accuracy', type=float, default=0.99)
    parser.add_argument('--min-accepted', type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.min_accepted < 1 or not 0 < args.target_accuracy <= 1:
        parser.error('epochs, patience, min-accepted must be positive; target-accuracy must be in (0,1]')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Output directory must be empty; choose a new --output to preserve previous runs')
    classes = CLASSES + [OTHER_CLASS] if args.include_other else CLASSES
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; all MobileNetV3-Small layers will be fine-tuned')
    splits, archive = prepare(args.data_dir, args.seed, include_other=args.include_other)
    (output / 'split_audit.json').write_text((Path(args.data_dir) / 'split_audit.json').read_text())
    (output / 'splits.json').write_text(json.dumps(splits, indent=2))
    counts = {s: {classes[c]: sum(r['label'] == c for r in rows) for c in range(len(classes))} for s, rows in splits.items()}
    print(json.dumps(counts, indent=2))
    batches = {s: loader(splits[s], image_transform(s == 'train'), archive,
                         args.batch_size, args.workers, s == 'train', args.seed) for s in ['train', 'validation']}
    model = build_model(pretrained=True, num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_counts = torch.tensor([counts['train'][c] for c in classes], dtype=torch.float, device=device)
    loss_weights = class_counts.sum() / (len(classes) * class_counts) if args.include_other else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    history, best_f1, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        progress = tqdm(batches['train'], desc=f'Epoch {epoch}/{args.epochs}')
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            progress.set_postfix(loss=loss_sum / count)
        y, prob = predict_loader(model, batches['validation'], device)
        score = float(f1_score(y, prob.argmax(1), labels=list(range(len(classes))), average='macro', zero_division=0))
        record = {'epoch': epoch, 'train_loss': loss_sum / count, 'validation_macro_f1': score,
                  'validation_accuracy': float((y == prob.argmax(1)).mean())}
        history.append(record)
        print(record)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if score > best_f1 + 1e-6:
            best_f1, stale = score, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'validation_macro_f1': score}, output / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping: validation macro F1 did not improve')
                break
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    y, prob = predict_loader(model, batches['validation'], device)
    selection = choose_threshold(y, prob, args.target_accuracy, args.min_accepted)
    (output / 'threshold_selection.json').write_text(json.dumps(selection, indent=2))
    metadata = {'schema_version': 1, 'architecture': 'mobilenet_v3_small', 'classes': classes,
                'class_to_index': {c: i for i, c in enumerate(classes)}, 'preprocessing': PREPROCESS,
                'reject_class_index': 3 if args.include_other else None,
                'loss_weights': loss_weights.cpu().tolist() if loss_weights is not None else None,
                'threshold': selection['threshold'], 'threshold_status': selection['status'],
                'dataset': 'mohanty/PlantVillage', 'configuration': 'color', 'revision': PV_REVISION,
                'arguments': vars(args), 'counts': counts, 'best_epoch': checkpoint['epoch'],
                'versions': {'python': platform.python_version(), **{name: version(name) for name in ['torch', 'torchvision', 'numpy', 'scikit-learn', 'Pillow', 'huggingface-hub', 'gradio']}}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_report(y, prob, selection['threshold'], output / 'reports', 'validation')
    print(f'Saved {output}; threshold={selection["threshold"]:.6f} ({selection["status"]})')
    print('No test images or PlantDoc images were used for selection. Run fieldready.evaluate next.')


if __name__ == '__main__':
    main()
