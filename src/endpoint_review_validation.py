"""Frozen retrospective selection sensitivity and input ablations.

See ieee_access/ENDPOINT_REVIEW_VALIDATION_PROTOCOL.md. No base retraining.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np
from basis_probe import columns
from channel_control import RollingControl


@dataclass(frozen=True)
class Config:
    name: str = 'proposed'
    inputs: tuple = ('z', 'a', 'r')
    k: int = 4
    ridge: float = 1.
    half_life: float | None = 128.
    window: int = 32
    summary: str = 'endpoint'


def configs():
    center = Config()
    out = [center]
    out += [replace(center, name=f'k{k}', k=k) for k in (1, 2, 3, 6, 8)]
    out += [replace(center, name=f'ridge{v:g}', ridge=v) for v in (.1, 10.)]
    out += [replace(center, name=f'half{v}', half_life=v) for v in (32, 512, None)]
    out += [replace(center, name=f'window{v}', window=v) for v in (8, 128)]
    assert len(out) == 13
    out += [replace(center, name=s, summary=s) for s in ('first', 'middle', 'mean')]
    out += [replace(center, name=name, inputs=inputs) for name, inputs in (
        ('no_endpoint', ('z', 'a')), ('endpoint_only', ('r',)),
        ('no_residual_dct', ('a', 'r')), ('no_forecast', ('z', 'r')))]
    out += [replace(center, name='persistence', inputs=('r',))]
    assert len(out) == 21
    return out


ABLATIONS = ['proposed', 'no_endpoint', 'endpoint_only',
             'no_residual_dct', 'no_forecast', 'persistence']


class PackedRidge:
    def __init__(self, k, c, p, ridge, half_life):
        self.p, self.ridge = p, ridge
        self.decay = 1. if half_life is None else 2. ** (-1. / half_life)
        self.s = np.zeros(1)
        self.cross = np.zeros((k, c, p))
        self.gram = np.zeros((k, c, p * (p + 1) // 2))
        self.rhs = np.zeros((k, c, p + 1))
        i, j = np.triu_indices(p)
        self.gram[..., i == j] = ridge

    def predict(self, v):
        i, j = np.triu_indices(self.p)
        mat = np.empty((*self.cross.shape[:-1], self.p, self.p))
        mat[..., i, j] = self.gram
        mat[..., j, i] = self.gram
        d = self.ridge + self.s[0]
        mat -= self.cross[..., :, None] * self.cross[..., None, :] / d
        rhs = self.rhs[..., 1:] - self.cross * self.rhs[..., 0, None] / d
        slope = np.linalg.solve(mat, rhs[..., None])[..., 0]
        intercept = (self.rhs[..., 0] - np.sum(self.cross * slope, axis=-1)) / d
        return intercept + np.sum(v * slope, axis=-1)

    def update(self, v, y):
        self.s *= self.decay
        self.s += 1
        self.cross *= self.decay
        self.cross += v
        self.gram *= self.decay
        i, j = np.triu_indices(self.p)
        self.gram += v[..., i] * v[..., j]
        self.gram[..., i == j] += (1 - self.decay) * self.ridge
        self.rhs *= self.decay
        self.rhs[..., 0] += y
        self.rhs[..., 1:] += v * y[..., None]

    @property
    def state_bytes(self):
        return sum(a.nbytes for a in (self.s, self.cross, self.gram, self.rhs))


class Tracker:
    def __init__(self, h, c, config):
        self.config, self.h = config, h
        self.last_r = np.zeros(c) if 'r' in config.inputs else None
        if config.name == 'persistence':
            return
        self.ids = np.arange(config.k, dtype=np.int64)
        self.u = columns(h, self.ids)
        self.last_z = np.zeros((config.k, c)) if 'z' in config.inputs else None
        self.model = PackedRidge(config.k, c, len(config.inputs), config.ridge, config.half_life)

    def issue(self, base):
        if self.config.name == 'persistence':
            return np.broadcast_to(self.last_r, base.shape), None
        values = []
        for feature in self.config.inputs:
            if feature == 'z': values.append(self.last_z)
            elif feature == 'a': values.append(self.u.T @ base)
            elif feature == 'r':
                values.append(np.broadcast_to(np.sqrt(self.h) * self.last_r,
                                              (self.config.k, base.shape[1])))
        v = np.stack(values, axis=-1)
        return self.u @ self.model.predict(v), v

    def update(self, residual, token):
        if self.config.name != 'persistence':
            z = self.u.T @ residual
            self.model.update(token, z)
            if self.last_z is not None: self.last_z = z
        if self.last_r is not None:
            if self.config.summary == 'first': value = residual[0]
            elif self.config.summary == 'middle': value = residual[(self.h - 1) // 2]
            elif self.config.summary == 'mean': value = residual.mean(axis=0)
            else: value = residual[-1]
            self.last_r = value.copy()

    @property
    def state_bytes(self):
        if self.config.name == 'persistence': return self.last_r.nbytes
        return (self.ids.nbytes + self.u.nbytes + self.model.state_bytes
                + (0 if self.last_z is None else self.last_z.nbytes)
                + (0 if self.last_r is None else self.last_r.nbytes))


def evaluate(base, target, config, predictions=False):
    base, target = np.asarray(base, float), np.asarray(target, float)
    assert base.shape == target.shape and base.ndim == 3
    n, h, c = base.shape
    tracker = Tracker(h, c, config)
    gate = RollingControl(c, 'global', 'blend', config.window)
    loss = np.empty((n, 2, 2))  # block, policy(raw/global), metric(MSE/MAE)
    alpha = np.empty(n)
    pred = np.empty_like(base) if predictions else None
    for t, (b, y) in enumerate(zip(base, target)):
        correction, token = tracker.issue(b)
        raw = b + correction
        # Match archived implementation's floating-point operation order.
        correction = raw - b
        alpha[t] = gate.alpha()[0]
        issued = b + alpha[t] * correction
        if predictions: pred[t] = issued
        for i, p in enumerate((raw, issued)):
            loss[t, i] = np.mean((p - y) ** 2), np.mean(abs(p - y))
        gate.update(correction, y - b)
        tracker.update(y - b, token)
    assert np.isfinite(loss).all()
    return dict(loss=loss, alpha=alpha, predictions=pred,
                raw_bytes=tracker.state_bytes,
                state_bytes=tracker.state_bytes + gate.state_bytes)


def intervals(n):
    half = n // 2
    return [('half', half, n), ('quarter2', n // 4, half),
            ('quarter3', half, 3 * n // 4), ('quarter4', 3 * n // 4, n)]


def select_prefix(losses, stop, candidate_ids):
    """Input shape: candidate, block, policy, metric; suffix is never read."""
    scores = losses[candidate_ids, :stop, 1, 0].mean(axis=1)
    return candidate_ids[int(np.argmin(scores))]
