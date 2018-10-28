from tests.data import train_data, test_data
from sdsj_feat import initial_processing


def test_run_train_test():
    df_train, params = initial_processing(train_data, mode='train')
    df_test, params = initial_processing(test_data, mode='train')


    assert True