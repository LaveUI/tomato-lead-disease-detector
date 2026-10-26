"""Synthetic integration checks, never reported as disease performance."""
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install requirements for runtime checks')
class RuntimeTests(unittest.TestCase):
    def test_rejection_predictions_remain_in_confusion_matrix(self):
        import numpy as np
        from fieldready.metrics import write_report
        with tempfile.TemporaryDirectory() as tmp:
            p = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 0, 1.]])
            report = write_report(np.array([0, 1, 2]), p, .9, tmp, 'fixture')
            self.assertEqual(sum(map(sum, report['confusion_matrix'])), 3)
            self.assertEqual(report['confusion_matrix'][2][3], 1)
            self.assertAlmostEqual(report['macro_f1'], 2 / 3)
            self.assertEqual(report['selective']['accepted'], 2)

    def test_four_class_checkpoint_always_rejects_other(self):
        import torch
        from PIL import Image
        from fieldready.data import CLASSES, OTHER_CLASS
        from fieldready.model import build_model, PREPROCESS
        from fieldready.inference import Predictor, UNCERTAIN
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = build_model(num_classes=4)
            with torch.no_grad():
                model.classifier[-1].weight.zero_()
                model.classifier[-1].bias.copy_(torch.tensor([0., 0., 0., 20.]))
            torch.save({'state_dict': model.state_dict()}, root / 'best.pt')
            (root / 'metadata.json').write_text(json.dumps({
                'classes': CLASSES + [OTHER_CLASS], 'architecture': 'mobilenet_v3_small',
                'threshold': 0., 'preprocessing': PREPROCESS}))
            result = Predictor(root).predict(Image.new('RGB', (224, 224)))
            self.assertFalse(result['accepted'])
            self.assertEqual(result['message'], UNCERTAIN)
            self.assertGreater(result['scores'][OTHER_CLASS], .99)

    def test_train_reload_evaluate_and_gradio(self):
        import torch
        from PIL import Image
        from fieldready import train, evaluate
        from fieldready.model import build_model
        from fieldready.inference import Predictor, UNCERTAIN
        import app
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'data.zip'
            splits = {}
            with zipfile.ZipFile(archive, 'w') as z:
                for split in ['train', 'validation', 'test']:
                    rows = []
                    for c in range(4):
                        for i in range(2):
                            name = f'{split}/{c}/{i}.jpg'
                            buf = io.BytesIO()
                            Image.new('RGB', (260, 260), (c * 80, i * 90, 30)).save(buf, format='JPEG')
                            z.writestr(name, buf.getvalue())
                            rows.append({'path': name, 'label': c, 'leaf_id': name})
                    splits[split] = rows
            (root / 'split_audit.json').write_text('{}')
            output = root / 'model'
            args = ['train', '--data-dir', str(root), '--output', str(output), '--epochs', '1',
                    '--batch-size', '4', '--workers', '0', '--include-other']
            with patch('sys.argv', args), patch.object(train, 'prepare', return_value=(splits, str(archive))), \
                 patch.object(train, 'build_model', side_effect=lambda pretrained, num_classes: build_model(False, num_classes)):
                train.main()
            predictor = Predictor(output)
            result = predictor.predict(Image.new('RGB', (300, 250)))
            self.assertEqual(result['message'], UNCERTAIN)  # min 30 exceeds synthetic validation size
            self.assertAlmostEqual(sum(result['scores'].values()), 1, places=5)
            self.assertEqual(predictor.predict(None)['scores'], {})
            # Build the real Gradio interface, suppress only server launch.
            with patch('sys.argv', ['app', '--model-dir', str(output)]), patch('gradio.Interface.launch') as launch:
                app.main()
                launch.assert_called_once()
            from fieldready.data import PD_MAP
            pd = root / 'PlantDoc'
            for split in ['train', 'test']:
                for folder in list(PD_MAP) + ['Tomato mold leaf']:
                    directory = pd / split / folder
                    directory.mkdir(parents=True)
                    Image.new('RGB', (256, 256)).save(directory / 'sample.jpg')
            with patch('sys.argv', ['evaluate', '--dataset', 'plantdoc', '--plantdoc', str(pd),
                                    '--model-dir', str(output), '--workers', '0']), \
                 patch('subprocess.check_output', return_value='synthetic-fixture'):
                evaluate.main()
            report = json.loads((output / 'reports/plantdoc_external.json').read_text())
            self.assertEqual(report['selective']['total'], 6)
            self.assertIsNone(report['selective']['accepted_accuracy'])
            self.assertTrue((output / 'reports/plantdoc_external_confusion.png').is_file())
            mold = json.loads((output / 'reports/plantdoc_leaf_mold_external.json').read_text())
            self.assertEqual(mold['selective']['total'], 2)
            self.assertEqual(mold['selective']['other_disease_rejection_rate'], 1)
