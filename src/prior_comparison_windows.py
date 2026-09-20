"""Causal windows for asymmetric-lookback residual training, not TEFL training itself."""
import numpy as np

def triplet(series,origin,lookback=96,horizon=24):
    """Past forecast origin is t-H; both contexts use the same lookback L."""
    if origin<lookback+horizon or origin+horizon>len(series):
        raise ValueError('insufficient context or target')
    return dict(previous_context=series[origin-horizon-lookback:origin-horizon].copy(),
                previous_target=series[origin-horizon:origin].copy(),
                current_context=series[origin-lookback:origin].copy(),
                current_target=series[origin:origin+horizon].copy())

def split_origins(start,end,lookback=96,horizon=24,stride=24):
    """Both context-target pairs entirely within [start,end); strict no-crossing policy."""
    if start<0 or end<start or stride<1:raise ValueError('invalid bounds')
    return np.arange(start+lookback+horizon,end-horizon+1,stride,dtype=np.int64)
