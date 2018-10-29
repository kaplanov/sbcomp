import argparse
import os
import pickle
import time
from profiler import Profiler
from sdsj_feat import load_data, initial_processing
from cat_transformer import CatTransformer

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)
    model = model_config['model']
    tf = model_config['cat_tf']
    train_params = model_config['params']
    # model_config['model'] = model
    # model_config['cat_tf'] = tf
    # model_config['params'] = params

    x_test_raw, _, _, df = load_data(args.test_csv, mode='test')

    # x_train, train_params = initial_processing(x_train_raw, mode='train')
    x_test, test_params = initial_processing(x_test_raw, mode='test')

    with Profiler('fit transform cat columns'):
        x_test_rein = x_test.reindex(columns=train_params['used_cols'])
        # tf = CatTransformer(train_params['cat_cols'])
        # tf.fit(x_train)
        # x_train_tf = tf.transform(x_train)
        x_test_tf = tf.transform(x_test_rein)

    # with Profiler('run train'):
    #     model = lgb.train(
    #         params,
    #         lgb.Dataset(x_train_tf, label=y_train),
    #         600)

    # df = pd.read_csv(args.test_csv, usecols=['line_id',])
    # print(args.test_csv)
    # df = pd.read_csv(args.test_csv)
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(x_test_tf)
    elif model_config['mode'] == 'classification':
        # df['prediction'] = model.predict_proba(X_scaled)[:, 1]
        df['prediction'] = model.predict(x_test_tf)

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {:0.2f}'.format(time.time() - start_time))
