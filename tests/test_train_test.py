from tests.data import train_data, test_data
from src.processing import initial_processing
from src.cat_transformer import CatTransformer


def test_run_train_test():
    df_train, train_params = initial_processing(train_data, mode='train')
    df_test, _ = initial_processing(test_data, mode='train')

    tf = CatTransformer(train_params['cat_cols'])
    tf.fit(df_train)
    df_train_tf = tf.transform(df_train)
    df_test_tf = tf.transform(df_test)

    assert set(df_train_tf.columns.values) == set(df_test_tf.columns.values)
