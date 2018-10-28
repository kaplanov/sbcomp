import pandas as pd
import numpy as np
from utils import transform_datetime_features
from profiler import Profiler

ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024


def get_mem(df):
    mem = df.memory_usage().sum() / 1000000
    return f'{mem:.2f}Mb'


def check_column_name(name):
    if name == 'line_id':
        return False
    if name.startswith('datetime'):
        return False
    if name.startswith('string'):
        return False
    if name.startswith('id'):
        return False

    return True


def load_test_label(path):
    y = pd.read_csv(path, low_memory=False).target
    return y


def load_data(path, mode='train', sample=None):
    with Profiler('read dataset'):
        is_big = False
        if mode == 'train':
            df = pd.read_csv(path, low_memory=False)
            shape_orig = df.shape
            if sample is not None:
                df = df.sample(n=sample)
            df.set_index('line_id', inplace=True)
            y = df.target
            df = df.drop('target', axis=1)
            if df.memory_usage().sum() > BIG_DATASET_SIZE:
                is_big = True
        else:
            df = pd.read_csv(path, low_memory=False)
            shape_orig = df.shape
            df.set_index('line_id', inplace=True)
            y = None

    print(f'Dataset read, orig: {shape_orig}, sampled: {df.shape}, memory: {get_mem(df)}, mode: {mode}')

    line_id = pd.DataFrame(df.index)

    return df, y, line_id


def initial_processing(df, mode):
    if df.memory_usage().sum() > BIG_DATASET_SIZE:
        is_big = True
    else:
        is_big = False

    with Profiler('features from datetime'):
        df, date_cols, orig_date_cols = transform_datetime_features(df)
    with Profiler('new cat'):
        cat_cols = get_cat_freqs(df)

    numeric_cols = [c for c in df.columns if c.startswith('number')]

    used_cols = date_cols + list(cat_cols) + numeric_cols
    df = df.reindex(columns=used_cols)

    # drop duplicate cols
    with Profiler('drop constant cols'):
        if mode == 'train':
            constant_columns = [
                col_name
                for col_name in df.columns
                if df[col_name].nunique() == 1
            ]
            print(f' - dropping {len(constant_columns)} columns')
            df.drop(constant_columns, axis=1, inplace=True)

    if is_big:
        df[numeric_cols] = df[numeric_cols].astype(np.float16)
    print(f' - Cat: {len(cat_cols)}, num: {len(numeric_cols)}, date: {len(date_cols)}, orig_dt: {len(orig_date_cols)}')
    print(f' - Used: {len(used_cols)}, memory: {get_mem(df)}')
    params = dict(
        cat_cols=cat_cols,
        numeric_cols=numeric_cols,
        date_cols=date_cols,
        used_cols=used_cols
    )
    return df, params


def get_cat_freqs(df):
    cat_cols = [col for col in df.columns.values if col.startswith('id') or col.startswith('string')]
    # freqs = {col: df[col].value_counts().to_dict() for col in cat_cols}

    return cat_cols


# def cat_transform(df, freq, upd_freqs=False):
#     if freq is None:
#         freq = {}
#
#     cat_cols, new_freq = get_cat_freqs(df)
#     if upd_freqs:
#         upd_freq = {**new_freq, **freq}
#     else:
#         upd_freq = {**freq}
#
#     for col in cat_cols:
#         df[col] = df[col].map(upd_freq[col])
#
#     df[cat_cols] = df[cat_cols].fillna(-1)
#
#     return df, upd_freq
