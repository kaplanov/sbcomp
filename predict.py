import argparse
import os
import pickle
import time
from profiler import Profiler
from processing import load_data, initial_processing
# from cat_transformer import CatTransformer

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
    train_params = model_config['train_params']

    x_test_raw, _, df = load_data(args.test_csv, mode='test')
    x_test, test_params = initial_processing(x_test_raw, mode='test')

    with Profiler('transform cat columns'):
        x_test_rein = x_test.reindex(columns=train_params['used_cols'])
        x_test_tf = tf.transform(x_test_rein)

    df['prediction'] = model.predict(x_test_tf)

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {:0.2f}'.format(time.time() - start_time))
