"""Evaluate a frozen model and threshold on untouched data."""
import argparse
import json
import subprocess
from pathlib import Path
from .data import plantdoc_rows, read_manifest, audit_splits, PV_REVISION
from .inference import Predictor
from .runtime import loader, predict_loader
from .metrics import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='artifacts/fieldready')
    parser.add_argument('--data-dir', default='data/plantvillage')
    parser.add_argument('--plantdoc', help='Path to the original cropped PlantDoc Git checkout')
    parser.add_argument('--dataset', choices=['plantvillage', 'plantdoc', 'both'], default='both')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.dataset in ['plantdoc', 'both'] and not args.plantdoc:
        parser.error('--plantdoc is required for external evaluation')
    predictor = Predictor(args.model_dir, args.device)
    output = Path(args.model_dir) / 'reports'
    selections = {}
    if args.dataset in ['plantvillage', 'both']:
        from huggingface_hub import hf_hub_download
        def fetch(name):
            return hf_hub_download('mohanty/PlantVillage', name, repo_type='dataset',
                                  revision=PV_REVISION, local_dir=args.data_dir)
        mapping = json.loads(Path(fetch('leaf_grouping/leaf-map.json')).read_text())
        official = read_manifest(fetch('splits/color_test.txt'), mapping, include_other=len(predictor.classes) == 4)
        saved = json.loads((Path(args.model_dir) / 'splits.json').read_text())
        if saved['test'] != official:
            raise ValueError('Saved test manifest differs from pinned official test split')
        audit_splits(saved)
        archive = fetch('data.zip')
        selections['plantvillage_test'] = ([r for r in official if r['label'] < 3], archive)
        if len(predictor.classes) == 4:
            selections['plantvillage_all_tomato_test'] = (official, archive)
            selections['plantvillage_leaf_mold_test'] = ([r for r in official if '/Tomato___Leaf_Mold/' in r['path']], archive)
    if args.dataset in ['plantdoc', 'both']:
        rows, inventory = plantdoc_rows(args.plantdoc, include_mold=len(predictor.classes) == 4)
        try:
            revision = subprocess.check_output(['git', '-C', args.plantdoc, 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = 'unknown: non-git dataset directory'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'plantdoc_inventory.json').write_text(json.dumps(
            {'revision': revision, 'folders': inventory, 'selected_images': rows,
             'protocol': 'Original train and test folders pooled as external evaluation only'}, indent=2))
        selections['plantdoc_external'] = ([r for r in rows if r['label'] < 3], None)
        if len(predictor.classes) == 4:
            selections['plantdoc_leaf_mold_external'] = ([r for r in rows if r['label'] == 3], None)
    for name, (rows, archive) in selections.items():
        batches = loader(rows, predictor.transform, archive, args.batch_size, args.workers)
        y, prob = predict_loader(predictor.model, batches, predictor.device)
        report = write_report(y, prob, predictor.threshold, output, name)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
