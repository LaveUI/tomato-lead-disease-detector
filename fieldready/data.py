"""Official color manifests, leaf grouping, and explicit external label mapping."""
import json
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ['Healthy', 'Early Blight', 'Late Blight']
PV_MAP = {'Tomato___healthy': 0, 'Tomato___Early_blight': 1, 'Tomato___Late_blight': 2}
OTHER_CLASS = 'Other tomato disease'
OTHER_PV = ['Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus']
PD_MAP = {'Tomato leaf': 0, 'Tomato Early blight leaf': 1, 'Tomato leaf late blight': 2}
PV_REVISION = '9e97599868962bd0079b8db4b7f1efa9185fa1e7'


def leaf_id(filename, class_name, mapping):
    # Match the pinned upstream plant_village.py identifier logic.
    identifier = filename.replace('_final_masked', '').split('___')[-1].split('copy')[0]
    for ext in ['.jpg', '.JPG', '.png', '.PNG']:
        identifier = identifier.replace(ext, '')
    identifier = identifier.strip()
    suggestions = mapping.get(identifier.lower(), [])
    if len(suggestions) == 1:
        return suggestions[0]
    return next((s for s in suggestions if class_name in s), f'fallback_{identifier}')


def read_manifest(path, mapping, include_other=False):
    labels = dict(PV_MAP)
    if include_other:
        labels.update({name: 3 for name in OTHER_PV})
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split('/')
        if len(parts) == 4 and parts[0:2] == ['raw', 'color'] and parts[2] in labels:
            rows.append({'path': line.strip(), 'label': labels[parts[2]],
                         'leaf_id': leaf_id(parts[3], parts[2], mapping)})
    if not rows or {r['label'] for r in rows} != set(labels.values()):
        raise ValueError('Manifest must contain all requested tomato categories')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate paths in supplied manifest')
    return rows


def split_training(rows, seed=42):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, val = next(splitter.split(rows, [r['label'] for r in rows], [r['leaf_id'] for r in rows]))
    groups = [[rows[i] for i in indices] for indices in (train, val)]
    audit_splits(dict(zip(['train', 'validation'], groups)))
    return groups


def audit_splits(splits):
    expected = {r['label'] for r in next(iter(splits.values()))}
    if not {0, 1, 2}.issubset(expected):
        raise ValueError('Missing required diagnosis class')
    names = list(splits)
    for name, rows in splits.items():
        if {r['label'] for r in rows} != expected:
            raise ValueError(f'{name} is missing a class')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ['path', 'leaf_id']:
                overlap = {r[key] for r in splits[a]} & {r[key] for r in splits[b]}
                if overlap:
                    raise ValueError(f'{a}/{b} overlap in {key}: {len(overlap)}; refusing leakage')


def plantdoc_rows(root, include_mold=False):
    labels = dict(PD_MAP)
    if include_mold:
        labels['Tomato mold leaf'] = 3
    root = Path(root)
    rows, inventory = [], {}
    for split in ['train', 'test']:
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing original PlantDoc directory: {directory}')
        inventory[split] = sorted(p.name for p in directory.iterdir() if p.is_dir())
        for name, label in labels.items():
            folder = directory / name
            if not folder.is_dir():
                raise ValueError(f'Expected exact folder missing: {folder}; inspect inventory {inventory}')
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    rows.append({'path': str(p), 'label': label, 'source_split': split})
    if {r['label'] for r in rows} != set(labels.values()):
        raise ValueError('PlantDoc selection is missing a class')
    return rows, inventory


def isolate_official_test(train, test):
    """Preserve every test image; explicitly quarantine overlapping training leaves."""
    test_ids = {r['leaf_id'] for r in test}
    excluded = [r for r in train if r['leaf_id'] in test_ids]
    eligible = [r for r in train if r['leaf_id'] not in test_ids]
    return eligible, excluded


def prepare(root, seed, include_other=False):
    from huggingface_hub import hf_hub_download
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        return hf_hub_download('mohanty/PlantVillage', name, repo_type='dataset',
                              revision=PV_REVISION, local_dir=root)
    mapping = json.loads(Path(fetch('leaf_grouping/leaf-map.json')).read_text())
    train = read_manifest(fetch('splits/color_train.txt'), mapping, include_other)
    test = read_manifest(fetch('splits/color_test.txt'), mapping, include_other)
    train, excluded = isolate_official_test(train, test)
    audit = {'official_train_selected': len(train) + len(excluded),
             'official_test_selected': len(test), 'quarantined_training_images': excluded,
             'reason': 'Training leaf_id also occurs in supplied test; test is unchanged'}
    (root / 'split_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'Quarantined {len(excluded)} official training images sharing test leaf IDs; see split_audit.json')
    train, validation = split_training(train, seed)
    splits = dict(train=train, validation=validation, test=test)
    audit_splits(splits)
    archive = fetch('data.zip')
    return splits, archive
