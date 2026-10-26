"""Recompute a decision threshold using only the saved validation partition."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .data import audit_splits
from .inference import Predictor
from .runtime import loader, predict_loader
from .metrics import choose_threshold, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='artifacts/fieldready')
    parser.add_argument('--data-dir', default='data/plantvillage')
    parser.add_argument('--target-accuracy', type=float, default=0.99)
    parser.add_argument('--min-accepted', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.target_accuracy <= 1 or args.min_accepted < 1:
        parser.error('target-accuracy must be in (0,1]; min-accepted must be positive')
    directory = Path(args.model_dir)
    predictor = Predictor(directory)
    splits = json.loads((directory / 'splits.json').read_text())
    audit_splits(splits)
    batches = loader(splits['validation'], predictor.transform,
                     str(Path(args.data_dir) / 'data.zip'), workers=args.workers)
    y, probabilities = predict_loader(predictor.model, batches, predictor.device)
    selection = choose_threshold(y, probabilities, args.target_accuracy, args.min_accepted)
    # Preserve the previous rule and reports. Model weights are never modified.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = directory / 'policy_backups' / stamp
    backup.mkdir(parents=True)
    for name in ['metadata.json', 'threshold_selection.json']:
        source = directory / name
        if source.exists():
            shutil.copy2(source, backup / name)
    if (directory / 'reports').exists():
        shutil.copytree(directory / 'reports', backup / 'reports')
    metadata = predictor.metadata
    metadata['threshold'] = selection['threshold']
    metadata['threshold_status'] = selection['status']
    metadata['recalibration'] = {'timestamp_utc': stamp, 'target_accuracy': args.target_accuracy,
                               'min_accepted': args.min_accepted, 'source': 'validation only',
                               'previous_policy_backup': str(backup)}
    (directory / 'threshold_selection.json').write_text(json.dumps(selection, indent=2))
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    report = write_report(y, probabilities, selection['threshold'], directory / 'reports', 'validation')
    print(json.dumps({'threshold': selection['threshold'], 'status': selection['status'],
                      'validation': report['selective'], 'backup': str(backup)}, indent=2))
    print('Restart the app to load this policy. Rerun test evaluation: existing test reports may use the old threshold.')


if __name__ == '__main__':
    main()
