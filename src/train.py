import argparse
import os
import pickle
import time
import lightgbm as lgb
from src.profiler import Profiler
from src.processing import load_data, initial_processing
from src.cat_transformer import CatTransformer
from src.feature_selection import ols_selection

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
_SAMPLE = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

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
    with Profiler('load data and perform feature selection'):
        x_ini_raw, y_ini, _ = load_data(args.train_csv, sample=_SAMPLE)
        x_initial, ini_params = initial_processing(x_ini_raw, mode='train')
        tf = CatTransformer(ini_params['cat_cols'])
        x_initial_tf = tf.fit_transform(x_initial)
        selected_features = ols_selection(x_initial_tf, y_ini)
    print(f'{ len(selected_features)} features selected')

    df_X_raw, df_y, _ = load_data(args.train_csv, used_cols=selected_features)
    x_train, train_params = initial_processing(df_X_raw, mode='train')

    with Profiler('fit transform cat columns'):
        tf = CatTransformer(train_params['cat_cols'])
        tf.fit(x_train)
        x_train_tf = tf.transform(x_train)

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
