from tests.data import train_data
from sdsj_feat import initial_processing


def test_cat_cols_frequency():
    df, params = initial_processing(train_data, mode='train')
    cat_cols = params['cat_freqs']

    assert set(cat_cols) == {'id_0', 'string_0', 'string_1'}



def test_date_col_processing():
    df, params = initial_processing(train_data, mode='train')

    assert df['date_month_datetime_0'][0] == 1
    assert df['date_month_datetime_0'][2] == 2
    assert df['date_month_datetime_0'][3] == 11

    assert df['date_weekday_datetime_0'][1] == 2
    assert df['date_weekday_datetime_0'][3] == 0
    assert df['date_weekday_datetime_0'][4] == 6

    assert df['date_day_datetime_0'][1] == 3
    assert df['date_day_datetime_0'][2] == 14
    assert df['date_day_datetime_0'][4] == 30
