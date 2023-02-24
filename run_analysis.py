import argparse
from ModelAnalysis import ModelAnalysis
from utils import *
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', nargs='+', help='options: nfl, income')
    parser.add_argument(
        '--models', nargs='+',
        help='models: decision_tree, knn, svm, ada_boost, mlp')
    args = parser.parse_args()

    dataset_list = args.datasets
    model_list = args.models

    model_eval_map = {
        "decision_tree":["Depth"],
        "knn": ["n_neighbors", "power"],
        "svm": ["C"],
        "ada_boost": ["N_estimators"],
        "mlp" : ["N_hidden_layers", "activation_func"]
    }

    model_range_map = {
        "Depth": np.arange(1, 30),
        "n_neighbors": np.arange(1, 505, 5),
        "C": [10**-3, 10**-2, 10**-1, 0.5, 1],
        "N_estimators": np.arange(30, 305, 5),
        "N_hidden_layers": [(32, ), (32, 16), (32, 16, 8), (32, 16, 8, 4)],
        "power": [2, 1],
        "min_samples_leaf": [2],
        "min_samples_split": [2],
        "activation_func": ["relu", "logistic"]
    }

    for dataset in dataset_list:
        X, y = get_data(dataset=dataset)

        for model_type in model_list:

            eval_types = model_eval_map[model_type]

            analysis = ModelAnalysis(X, y, model_type=model_type)
            analysis.set_train_test_split()

            for etype in eval_types:
                analysis.eval_overfit(model_range_map[etype], eval_type=etype)
                for key in analysis.test_eval_metrics:
                    eval_type = key[0]
                    eval_metric = key[1]

                    filepath = "plots/{}/{}_{}_{}".format(
                        dataset, model_type, eval_type, eval_metric)
                    analysis.plot_overfit(eval_type, eval_metric, dataset, filepath)

                n = analysis.get_best_model_testing(eval_type=eval_type, eval_metric="AUC")

                print("The optimal parameter for {} is {}".format(eval_type, n))

            analysis.plot_learning_curves(
                n=5, scoring_metric_list=['accuracy'], dataset=dataset)
            analysis.save_classification_report(dataset)


if __name__ == '__main__':
    main()
