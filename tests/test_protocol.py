import unittest
import numpy as np
from fieldready.data import split_training, audit_splits, leaf_id, isolate_official_test
from fieldready.metrics import choose_threshold, selective_metrics


class ProtocolTests(unittest.TestCase):
    def test_grouped_split_reproducible(self):
        rows = [{'path': f'{c}/{g}/{i}', 'label': c, 'leaf_id': f'{c}/{g}'}
                for c in range(3) for g in range(20) for i in range(2)]
        a, b = split_training(rows)
        self.assertEqual((a, b), tuple(split_training(rows)))
        self.assertEqual(len(a) + len(b), len(rows))
        self.assertFalse({r['leaf_id'] for r in a} & {r['leaf_id'] for r in b})

    def test_overlap_rejected(self):
        rows = [{'path': str(c), 'label': c, 'leaf_id': str(c)} for c in range(3)]
        with self.assertRaises(ValueError):
            audit_splits({'train': rows, 'test': rows})

    def test_quarantine_preserves_test(self):
        train = [{'path': 'a', 'leaf_id': 'shared'}, {'path': 'b', 'leaf_id': 'unique'}]
        test = [{'path': 'c', 'leaf_id': 'shared'}]
        eligible, excluded = isolate_official_test(train, test)
        self.assertEqual(eligible, [train[1]])
        self.assertEqual(excluded, [train[0]])
        self.assertEqual(test, [{'path': 'c', 'leaf_id': 'shared'}])

    def test_leaf_aliases(self):
        self.assertEqual(leaf_id('uuid___LEAF copy.JPG', 'Tomato', {'leaf': ['id']}), 'id')
        self.assertEqual(leaf_id('uuid___abc.jpg', 'Tomato', {}), 'fallback_abc')

    def test_threshold_excludes_wrong_low_scores(self):
        y = np.array([0, 1, 2])
        p = np.array([[.9, .05, .05], [.1, .8, .1], [.4, .35, .25]])
        result = choose_threshold(y, p, target_accuracy=1, min_accepted=2)
        selected = selective_metrics(y, p, result['threshold'])
        self.assertEqual(selected['accepted'], 2)
        self.assertEqual(selected['accepted_accuracy'], 1)

    def test_easy_validation_does_not_disable_abstention(self):
        y = np.array([0, 1, 2])
        p = np.array([[.9, .05, .05], [.1, .8, .1], [.1, .2, .7]])
        result = choose_threshold(y, p, target_accuracy=.99, min_accepted=1)
        self.assertAlmostEqual(result['threshold'], .7)
        low_confidence = np.array([[.34, .33, .33]])
        self.assertEqual(selective_metrics(np.array([0]), low_confidence, result['threshold'])['accepted'], 0)

    def test_other_class_never_accepted_and_errors_count(self):
        y = np.array([3, 3, 0])
        p = np.array([[0., 0., 0., 1.], [.9, 0., 0., .1], [.9, .1, 0., 0.]])
        r = selective_metrics(y, p, .8)
        self.assertEqual(r['accepted'], 2)
        self.assertEqual(r['accepted_accuracy'], .5)
        self.assertEqual(r['other_disease_rejection_rate'], .5)
        policy = choose_threshold(y, p, target_accuracy=1, min_accepted=1)
        self.assertGreater(policy['threshold'], 1)

    def test_no_feasible_threshold_and_ties(self):
        y = np.array([1, 0])
        p = np.array([[1., 0, 0], [1., 0, 0]])
        result = choose_threshold(y, p, target_accuracy=1, min_accepted=1)
        self.assertGreater(result['threshold'], 1)
        self.assertIsNone(selective_metrics(y, p, result['threshold'])['accepted_accuracy'])


if __name__ == '__main__':
    unittest.main()
