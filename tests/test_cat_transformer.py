import numpy as np
from tests.data import train_data, test_data
from cat_transformer import CatTransformer
from processing import initial_processing


def test_cat_fit_train():
    _, params = initial_processing(train_data, mode='train')
    cat_cols = params['cat_cols']

    tf = CatTransformer(cat_cols)
    tf.fit(train_data[cat_cols])
    res_df = tf.transform(train_data)
    assert res_df['string_0'][0] == 2 / 5
    assert res_df['string_0'][1] == 1 / 5
    assert res_df['string_0'][2] == 2 / 5
    assert np.isnan(res_df['string_0'][3])
    assert res_df['string_0'][4] == 1 / 5

    assert res_df['string_1'][0] == 2 / 5
    assert np.isnan(res_df['string_1'][1])
    assert res_df['string_1'][2] == 1 / 5
    assert res_df['string_1'][3] == 2 / 5
    assert res_df['string_1'][4] == 1 / 5


def test_cat_fit_test():
    _, params = initial_processing(train_data, mode='train')
    cat_cols = params['cat_cols']

    tf = CatTransformer(cat_cols)
    tf.fit(train_data[cat_cols])
    res_df = tf.transform(test_data)
    assert res_df['string_0'][0] == 2 / 5
    assert res_df['string_0'][1] == 1 / 5
    assert res_df['string_0'][2] == 2 / 5
    assert np.isnan(res_df['string_0'][3])
    assert np.isnan(res_df['string_0'][4])

    assert np.isnan(res_df['string_1']).all()
