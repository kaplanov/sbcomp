import pandas as pd
from scipy.stats import linregress
import lightgbm as lgb
from profiler import Profiler

_r2_thresh = 0.001


def _extract_orig_col(col):
    if col.startswith('date'):
        if col[-10] == 'd':
            return col[-10:]
        else:
            return col[-11:]
    else:
        return col


def ols_selection(df, y, objective):
    with Profiler('lgb selection'):
        feat_lgb = calc_lgb(df, y, objective)
        feat_dict = feat_lgb['gain'].to_dict().items()

    with Profiler('ols selection'):
        features = calc_ols(df, y)
        r2_dict = features.T['r_squared'].to_dict().items()

    feat_list = [_extract_orig_col(col) for col, r2 in feat_dict if r2 > _r2_thresh]
    unique_features = set(feat_list)

    res = list(unique_features)
    return res


def calc_ols(df, y):
    res = {}
    for col in df.columns.values:
        _, _, r_value, p_value, std_err = linregress(df[col].fillna(-1), y)
        res[col] = {
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

    return pd.DataFrame(res)


def calc_lgb(x, y, objective):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': 'rmse',
        "verbosity": -1
    }

    model = lgb.train(
        params,
        lgb.Dataset(x, label=y),
        600)
    gain = pd.Series(model.feature_importance(importance_type='gain'), index=x.columns.values)
    gain /= gain.sum()
    gain.name = 'gain'

    split = pd.Series(model.feature_importance(importance_type='split'), index=x.columns.values)
    split /= split.sum()
    split.name = 'split'
    return pd.DataFrame([gain, split]).T
