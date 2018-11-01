import pandas as pd
from scipy.stats import linregress

_r2_thresh = 0.005


def _extract_orig_col(col):
    if col.startswith('date'):
        if col[-10] == 'd':
            return col[-10:]
        else:
            return col[-11:]
    else:
        return col


def ols_selection(df, y):
    features = calc_ols(df, y)
    r2_dict = features.T['r_squared'].to_dict().items()

    feat_list = [_extract_orig_col(col) for col, r2 in r2_dict if r2 > _r2_thresh]
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