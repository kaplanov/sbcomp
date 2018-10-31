import argparse
import os
import pickle
import time
import lightgbm as lgb
from src.profiler import Profiler
from src.processing import load_data, initial_processing
from src.cat_transformer import CatTransformer

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df_X_raw, df_y, _ = load_data(args.train_csv)

    model_config = dict()
    model_config['mode'] = args.mode

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression' if args.mode == 'regression' else 'binary',
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
    x_train, train_params = initial_processing(df_X_raw, mode='train')
    # x_test, test_params = initial_processing(x_test_raw, mode='test')

    with Profiler('fit transform cat columns'):
        # x_test_rein = x_test.reindex(columns=train_params['used_cols'])
        tf = CatTransformer(train_params['cat_cols'])
        tf.fit(x_train)
        x_train_tf = tf.transform(x_train)
        # x_test_tf = tf.transform(x_test_rein)

    with Profiler('run train'):
        model = lgb.train(
            params,
            lgb.Dataset(x_train_tf, label=df_y),
            600)

    model_config['model'] = model
    model_config['cat_tf'] = tf
    model_config['params'] = params
    model_config['train_params'] = train_params

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {:0.2f}'.format(time.time() - start_time))
