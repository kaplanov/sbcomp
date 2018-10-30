import numpy as np
import lightgbm as lgb
from sdsj_feat import load_data, load_test_label, initial_processing
from sklearn.metrics import mean_squared_error, roc_auc_score
from cat_transformer import CatTransformer
from profiler import Profiler

_DATA_PATH = 'data/'

data_sets = [
    'check_1_r', 'check_2_r', 'check_3_r',
    'check_4_c', 'check_5_c', 'check_6_c',
    'check_7_c',
    'check_8_c'
]


def run_train_test(ds_name, metric, params, sample_train):
    path = _DATA_PATH + ds_name
    x_train_raw, y_train, _ = load_data(f'{path}/train.csv', mode='train', sample=sample_train)
    x_test_raw, _, _ = load_data(f'{path}/test.csv', mode='test')
    y_test = load_test_label(f'{path}/test-target.csv')

    x_train, train_params = initial_processing(x_train_raw, mode='train')
    x_test, test_params = initial_processing(x_test_raw, mode='test')

    with Profiler('fit transform cat columns'):
        x_test_rein = x_test.reindex(columns=train_params['used_cols'])
        tf = CatTransformer(train_params['cat_cols'])
        tf.fit(x_train)
        x_train_tf = tf.transform(x_train)
        x_test_tf = tf.transform(x_test_rein)

    with Profiler('run train'):
        model = lgb.train(
            params,
            lgb.Dataset(x_train_tf, label=y_train),
            600)

    with Profiler('predict'):
        y_train_out = model.predict(x_train_tf)
        y_test_out = model.predict(x_test_tf)

    train_err = metric(y_train, y_train_out)
    test_err = metric(y_test, y_test_out)

    return train_err, test_err


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    for data_path in data_sets:
        mode = data_path[-1]
        default_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression' if mode == 'r' else 'binary',
            'metric': 'rmse',
            "learning_rate": 0.01,
            "num_leaves": 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            'bagging_freq': 4,
            "max_depth": -1,
            "verbosity": -1,
            "reg_alpha": 0.3,
            "reg_lambda": 0.1,
            "min_child_weight": 10,
            'zero_as_missing': True,
            'num_threads': 4,
            'seed': 1
        }
        metric = roc_auc_score if mode == 'c' else rmse
        mt_name = 'auc' if mode == 'c' else 'rmse'
        train_err, test_err = run_train_test(data_path, metric, default_params, 10000)

        print(f'ds={data_path} train_{mt_name}={train_err:.4f} test_{mt_name}={test_err:.4f}')
        print()


if __name__ == '__main__':
    main()
