import pandas as pd
from sdsj_feat import initial_processing




def test_cat_cols_frequency():
    df, cat_cols, numeric_cols, date_cols, used_columns = initial_processing(train_data, mode='train')

    assert set(cat_cols.keys()) == {'id_0', 'string_0', 'string_1'}
    assert cat_cols['string_1']['val_2'] == 1
    assert cat_cols['string_1']['val_3'] == 2
    assert cat_cols['string_1']['val_4'] == 1

    assert cat_cols['string_0']['val_1'] == 2
    assert cat_cols['string_0']['val_2'] == 1
    assert cat_cols['string_0']['val_3'] == 1