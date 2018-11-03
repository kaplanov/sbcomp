import numpy as np
import lightgbm as lgb
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials, fmin
from sklearn.model_selection import train_test_split


# def train_lightgbm(x, y, config: Config):
#     params = {
#         "objective": "regression" if config["mode"] == "regression" else "binary",
#         "metric": "rmse" if config["mode"] == "regression" else "auc",
#         "verbosity": -1,
#         "seed": 1,
#     }
#
#     X_sample, y_sample = data_sample(X, y)
#     hyperparams = hyperopt_lgb(X_sample, y_sample, params, config)
#
#     X_train, X_val, y_train, y_val = data_split(X, y)
#     train_data = lgb.Dataset(X_train, label=y_train)
#     valid_data = lgb.Dataset(X_val, label=y_val)
#
#     config["model"] = lgb.train({**params, **hyperparams}, train_data, 3000, valid_data, early_stopping_rounds=50,
#                                 verbose_eval=100)


def hyperopt_lgb(x, y, params, obj):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=30, verbose_eval=100)

        score = model.best_score["valid_0"][params["metric"]]
        if obj == 'binary':
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=50, verbose=1,
                rstate=np.random.RandomState(1))

    hyper_params = space_eval(space, best)
    print("{:0.4f} {}".format(trials.best_trial['result']['loss'], hyper_params))
    return hyper_params
