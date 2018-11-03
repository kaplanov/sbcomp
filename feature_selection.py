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
    # with Profiler('lgb selection'):
    #     feat_lgb = calc_lgb(df, y, objective)
    #     feat_dict = feat_lgb['gain'].to_dict().items()

    with Profiler('ols selection'):
        features = calc_ols(df, y)
        r2_dict = features.T['r_squared'].to_dict()
        var_dict = features.T['var'].to_dict()

    tf_features = [col for col, r2 in r2_dict.items() if r2 > _r2_thresh and var_dict[col] != 0]
    feat_list = [_extract_orig_col(col) for col in tf_features]
    unique_features = set(feat_list)

    res = list(unique_features)
    return res, tf_features


def calc_ols(df, y):
    res = {}
    for col in df.columns.values:
        _, _, r_value, p_value, std_err = linregress(df[col].fillna(-1), y)
        var = df[col].var()
        res[col] = {
            'r_value': r_value,
            'r_squared': r_value ** 2,
            'var': var
        }

    return pd.DataFrame(res)


def calc_lgb(x, y, objective):
    params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 200,
        'subsample': 1,
        'colsample_bytree': 1,
        'random_state': 42,
        'n_jobs': -1
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
