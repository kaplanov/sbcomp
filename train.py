import argparse
import os
import pickle
import time
import lightgbm as lgb
from sdsj_feat import load_data
from sklearn.preprocessing import LabelEncoder

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

ONEHOT_MAX_UNIQUE_VALUES = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df_X, df_y, model_config, _ = load_data(args.train_csv)

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

    model = lgb.train(params, lgb.Dataset(df_X, label=df_y), 600)
    label_enc = LabelEncoder()
    cat_features = model_config['categorical_values']
    print('cat features=', cat_features)
    label_enc.fit(df_X[cat_features])
    df_X = label_enc.transform(df_X[cat_features])

    model_config['model'] = model
    model_config['label_enc'] = label_enc
    model_config['params'] = params

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
