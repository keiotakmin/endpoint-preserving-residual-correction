import unittest
import numpy as np
from endpoint_review_validation import Config, PackedRidge, configs, evaluate, select_prefix
from boundary_context import evaluate as old_evaluate
from feature_position import evaluate as feature_evaluate


class ValidationTests(unittest.TestCase):
    def test_weighted_augmented_least_squares(self):
        rng = np.random.default_rng(912)
        for p in (1, 2, 3):
            v = rng.normal(size=(18, 2, 3, p))
            y = rng.normal(size=(18, 2, 3))
            x = np.concatenate((np.ones((*v.shape[:-1], 1)), v), axis=-1)
            for ridge in (.1, 1., 10.):
                for half in (4., 128., None):
                    model = PackedRidge(2, 3, p, ridge, half)
                    for t in range(18):
                        expected = np.zeros((2, 3))
                        if t:
                            w = model.decay ** np.arange(t - 1, -1, -1)
                            # Independent square-root-weighted augmented lstsq, not Gram recursion.
                            for j in range(2):
                                for c in range(3):
                                    design = np.concatenate((x[:t, j, c] * np.sqrt(w[:, None]),
                                                             np.sqrt(ridge) * np.eye(p + 1)))
                                    response = np.r_[y[:t, j, c] * np.sqrt(w), np.zeros(p + 1)]
                                    expected[j, c] = x[t, j, c] @ np.linalg.lstsq(design, response, rcond=None)[0]
                        np.testing.assert_allclose(model.predict(v[t]), expected, atol=1e-11, rtol=1e-11)
                        model.update(v[t], y[t])

    def test_archived_equivalence_and_sizes(self):
        rng = np.random.default_rng(21)
        b = rng.normal(size=(85, 24, 7))
        y = b + rng.normal(size=b.shape)
        mapping = {'proposed': 'boundary4', 'no_endpoint': 'independent4', 'persistence': 'last_residual'}
        for cfg in configs():
            if cfg.name in mapping:
                old = old_evaluate(b, y, mapping[cfg.name], trace=True)
                size = old['learner_basis_bytes'] + old['gate_bytes']
            elif cfg.name in ('first', 'middle', 'mean'):
                old = feature_evaluate(b, y, cfg.name, trace=True)
                size = old['state_bytes']
            else: continue
            new = evaluate(b, y, cfg, predictions=True)
            np.testing.assert_allclose(new['predictions'], old['predictions'], rtol=0, atol=1e-11)
            self.assertEqual(new['state_bytes'], size)
        self.assertEqual(evaluate(b, y, Config())['state_bytes'], 4528)
        sizes = {c.name: evaluate(b[:2], y[:2], c)['state_bytes'] for c in configs()}
        # p slopes store p(p+1)/2 + p + (p+1) entries per component/channel.
        self.assertEqual(sizes['endpoint_only'], 2288)
        self.assertEqual(sizes['no_residual_dct'], 3184)
        self.assertEqual(sizes['no_forecast'], 3408)
        self.assertEqual(sizes['persistence'], 584)

    def test_future_targets_cannot_change_issued_prediction_or_selection(self):
        rng = np.random.default_rng(11)
        b = rng.normal(size=(30, 24, 2)); y = rng.normal(size=b.shape)
        changed = y.copy(); changed[15:] += 100
        one, two = [], []
        for cfg in configs():
            a = evaluate(b, y, cfg, predictions=True)
            z = evaluate(b, changed, cfg, predictions=True)
            np.testing.assert_array_equal(a['predictions'][:16], z['predictions'][:16])
            one.append(a['loss']); two.append(z['loss'])
        for ids in (list(range(13)), list(range(21))):
            self.assertEqual(select_prefix(np.array(one), 15, ids), select_prefix(np.array(two), 15, ids))


if __name__ == '__main__': unittest.main()
