import pandas as pd

train_data = pd.DataFrame({
    'string_0': ['val_1', 'val_3', 'val_1', None, 'val_2'],
    'string_1': ['val_3', None, 'val_4', 'val_3', 'val_2'],
    'number_0': [1, 2, 3, 4, 5],
    'number_1': [1, 2, 3, 4, 5],
    'number_2': [1, 1, 1, 1, 1],
    'number_3': [None, None, 1, None, 4],
    'id_0': ['val_1', 'val_2', 'val_3', 'val_4', 'val_5'],
    'datetime_0': ['2018-01-01', '2018-01-03', '2018-02-14', '2018-11-05', '2018-12-30']
})


test_data = pd.DataFrame({
    'string_0': ['val_1', 'val_3', 'val_1', None, 'val_5'],
    'string_1': ['val_8', None, 'val_8', 'val_8', 'val_8'],
    'number_0': [1, 2, 3, 4, 5],
    'number_1': [1, 2, 3, 4, 5],
    'number_2': [1, 1, 1, 1, 1],
    'number_3': [None, None, 1, None, 4],
    'id_0': ['val_1', 'val_2', 'val_3', 'val_4', 'val_5'],
    'datetime_0': ['2018-01-01', '2018-01-03', '2018-02-14', '2018-11-05', '2018-12-30']
})
