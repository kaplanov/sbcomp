def _extract_orig_col(col):
    if col.startswith('date'):
        if col[-10] == 'd':
            return col[-10:]
        else:
            return col[-11:]
    else:
        return col


def ols_selection(df):
    feat_list = [_extract_orig_col(c) for c in df.columns.values]
    uniq = set(feat_list)

    res = list(uniq)
    return res
