import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .data import CLASSES, OTHER_CLASS


def selective_metrics(y, probabilities, threshold):
    accepted = (probabilities.argmax(axis=1) < 3) & (probabilities.max(axis=1) >= threshold)
    n = int(accepted.sum())
    return {'total': len(y), 'accepted': n, 'abstained': len(y) - n,
            'coverage': float(accepted.mean()),
            'accepted_accuracy': float((probabilities.argmax(1)[accepted] == y[accepted]).mean()) if n else None,
            'other_disease_total': int((y == 3).sum()),
            'other_disease_rejection_rate': float((~accepted[y == 3]).mean()) if (y == 3).any() else None}


def choose_threshold(y, probabilities, target_accuracy=0.99, min_accepted=30):
    candidates = sorted(set(probabilities.max(1).tolist()))
    curve = [dict(threshold=float(t), **selective_metrics(y, probabilities, t)) for t in candidates]
    feasible = [r for r in curve if r['accepted'] >= min_accepted and r['accepted_accuracy'] >= target_accuracy]
    best = max(feasible, key=lambda r: (r['coverage'], -r['threshold'])) if feasible else None
    return {'threshold': best['threshold'] if best else 1.000001,
            'status': 'target_met' if best else 'no_feasible_threshold_abstain_all',
            'target_accuracy': target_accuracy, 'min_accepted': min_accepted,
            'selection_data': 'PlantVillage validation only', 'curve': curve}


def write_report(y, probabilities, threshold, output, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = probabilities.argmax(1)
    classes = CLASSES + [OTHER_CLASS] if probabilities.shape[1] == 4 else CLASSES
    labels = list(range(len(classes)))
    metric_labels = sorted(set(y.tolist()))
    matrix = confusion_matrix(y, pred, labels=labels)
    per_class = classification_report(y, pred, labels=labels, target_names=classes,
                                      output_dict=True, zero_division=0)
    report = {'dataset': name, 'accuracy': float(accuracy_score(y, pred)),
              'macro_f1': float(f1_score(y, pred, labels=metric_labels, average='macro', zero_division=0)),
              'macro_f1_classes': [classes[i] for i in metric_labels],
              'per_class': {c: per_class[c] for c in classes},
              'confusion_matrix': matrix.tolist(), 'class_order': classes, 'threshold': threshold,
              'selective': selective_metrics(y, probabilities, threshold)}
    (output / f'{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / f'{name}_predictions.npz', targets=y, probabilities=probabilities)
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(cmap='Blues', values_format='d')
    plt.tight_layout()
    plt.savefig(output / f'{name}_confusion.png', dpi=160)
    plt.close()
    return report
