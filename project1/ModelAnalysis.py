import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from typing import Union

from utils import *
import numpy as np

class ModelAnalysis(object):
    """Model Analysis object takes in a X dataset and Y binary predictors.
    """
    def __init__(self, X: pd.DataFrame, y: Union[list, np.ndarray], model_type: str):
        """
        Params:
            X: m x n dataframe of inputs, numerics only.
            y: m x 1 output of binary classifcation [0 or 1]
            model_type:
                - decision_tree
                - knn
                - svm
                - ada_boost
                - mlp
        """
        self.X = X
        self.y = y
        self.model_type = model_type

        self.train_eval_metrics = {}
        self.test_eval_metrics = {}
        self.eval_range = {}
        self.best_model = None

        # Static params.
        self.metrics_list = [
            "MSE", "AUC", "Precision", "Recall", "F1", "Loss", "Accuracy"]

        # Tree params
        self.tree_params = ["Depth", "min_sample_split", "min_samples_leaf"]
        self.knn_params = ["n_neighbors", "power"]
        self.mlp_params = ["N_hidden_layers", "activation_func"]


    def set_train_test_split(self, test_size: float=0.3, seed: int=42):
        """This method sets the test_size to split the X and y values into
        train and testing.

        Params:
            test_size: split on the X and y for testing
            seed: seed to reproduce results
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=seed
        )

    def _get_model(self, i, eval_type):
        """This method gets the model depending on the eval_type.

        Params:
            i: is the range where the hyperparameter is iterated for optimization
            eval_type: check the eval map for possible paramters
        """

        if self.model_type == "decision_tree":
            if self.best_model is None:
                model = tree.DecisionTreeClassifier(
                    random_state=42)
            else:
                model = self.best_model
            if eval_type not in self.tree_params:
                raise ValueError("Wrong eval type.")
            if eval_type == "Depth":
                model.max_depth = i
            if eval_type == "min_samples_leaf":
                model.min_samples_leaf = i
            if eval_type == "min_sample_split":
                model.min_samples_split = i
            return model

        if self.model_type == "knn":
            if self.best_model is None:
                model = KNeighborsClassifier()
            else:
                model = self.best_model

            if eval_type not in self.knn_params:
                raise ValueError("Wrong eval type.")
            if eval_type =="n_neighbors":
                model.n_neighbors = i
            if eval_type =="p":
                model.p = i
            return model

        if self.model_type == "svm":
            if eval_type =="C":
                model = SVC(C=i, kernel="linear")
                return model
            else:
                raise ValueError("Wrong eval type.")

        elif self.model_type == "ada_boost":
            base_tree = tree.DecisionTreeClassifier(
                    random_state=42, max_depth=1)
            if eval_type == "N_estimators":
                model = AdaBoostClassifier(
                    estimator=base_tree, n_estimators=i, learning_rate=1)
                return model
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "mlp":
            if self.best_model is None:
                model = MLPClassifier(
                    alpha=0.0001,
                    solver="adam",
                    max_iter=200)
            else:
                model = self.best_model
            if eval_type not in self.mlp_params:
                raise ValueError("Wrong eval type.")
            if eval_type == "N_hidden_layers":
                model.hidden_layer_sizes = i
            if eval_type == "activation_func":
                model.activation = i
            return model
        else:
            raise ValueError("Model type does not exist.")

    def eval_overfit(self, eval_range, eval_type):
        """This method takes an eval_range and eval_type to evaluate
        overfitting on the model.
        """
        # Initialize lists for metric values.
        for metric in self.metrics_list:
            key = (eval_type, metric)
            self.train_eval_metrics[key] = []
            self.test_eval_metrics[key] = []

        for i in eval_range:

            model = self._get_model(i, eval_type)

            if model == None:
                raise ValueError("Model type does not exist.")

            print("Training model for iteration {}".format(i))
            model = model.fit(self.X_train, self.y_train)

            y_test_pred = model.predict(self.X_test)
            y_train_pred = model.predict(self.X_train)

            # Calculate MSE though it is not a great metric for classification.
            mse_test = mean_squared_error(self.y_test, y_test_pred)
            mse_train = mean_squared_error(self.y_train, y_train_pred)

            self.train_eval_metrics[eval_type, "MSE"].append(mse_train)
            self.test_eval_metrics[eval_type, "MSE"].append(mse_test)

            # Calculate fpr, tpr to generate AUC.
            # AUC is the measure of FPR to TPR at various thresholds.
            fpr, tpr, thresholds = roc_curve(self.y_train, y_train_pred)
            auc_train = auc(fpr, tpr)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred)
            auc_test = auc(fpr, tpr)

            self.train_eval_metrics[eval_type, "AUC"].append(auc_train)
            self.test_eval_metrics[eval_type, "AUC"].append(auc_test)

            # Precision score.
            precision_train = precision_score(self.y_train, y_train_pred)
            precision_test = precision_score(self.y_test, y_test_pred)

            self.train_eval_metrics[eval_type, "Precision"].append(precision_train)
            self.test_eval_metrics[eval_type, "Precision"].append(precision_test)

            # Recall score.
            recall_train = recall_score(self.y_train, y_train_pred)
            recall_test = recall_score(self.y_test, y_test_pred)

            self.train_eval_metrics[eval_type, "Recall"].append(recall_train)
            self.test_eval_metrics[eval_type, "Recall"].append(recall_test)

            # F1 score
            f1_train = f1_score(self.y_train, y_train_pred)
            f1_test = f1_score(self.y_test, y_test_pred)

            self.train_eval_metrics[eval_type, "F1"].append(f1_train)
            self.test_eval_metrics[eval_type, "F1"].append(f1_test)

            # Loss
            loss_train = log_loss(self.y_train, y_train_pred)
            loss_test = log_loss(self.y_test, y_test_pred)

            self.train_eval_metrics[eval_type, "Loss"].append(loss_train)
            self.test_eval_metrics[eval_type, "Loss"].append(loss_test)

            # Accuracy
            accuracy_train = accuracy_score(self.y_train, y_train_pred)
            accuracy_test = accuracy_score(self.y_test, y_test_pred)

            self.train_eval_metrics[eval_type, "Accuracy"].append(accuracy_train)
            self.test_eval_metrics[eval_type, "Accuracy"].append(accuracy_test)


        self.eval_range[eval_type] = eval_range

        print("Finished evaluating model.")

    def plot_overfit(self, eval_type, eval_metric, dataset, filepath):
        """This method takes in an eval_type and an eval_metric to plot
        overfitting.
        """
        try:
            test = self.test_eval_metrics[eval_type, eval_metric]
            train = self.train_eval_metrics[eval_type, eval_metric]

            if eval_type == "N_hidden_layers":
                eval_range = np.arange(1, len(self.eval_range[eval_type])+1)
                print("The eval_range is {}".format(eval_range))
            else:
                eval_range = self.eval_range[eval_type]
        except KeyError:
            raise ValueError("Wrong eval metric.")

        fig = plt.figure()

        model_name = self.model_type.title().replace("_", " ")
        plt.title("{} - {} {}".format(dataset.upper(), model_name, eval_metric))
        plt.plot(eval_range, train, label="Train")
        plt.plot(eval_range, test, label="Validation")
        plt.legend()
        plt.xlabel("{}".format(eval_type))
        plt.ylabel(eval_metric)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(filepath, dpi=300)
        # plt.show()


    def get_best_model_testing(self,
                               eval_type="Depth",
                               eval_metric="AUC",
                               seed=42):

        test = self.test_eval_metrics[eval_type, eval_metric]
        n = self.eval_range[eval_type][np.argmax(test)]

        if self.model_type == "decision_tree":
            if self.best_model is None:
                self.best_model = tree.DecisionTreeClassifier(
                        random_state=42)
            if eval_type not in self.tree_params:
                raise ValueError("Wrong eval type.")
            if eval_type == "Depth":
                self.best_model.max_depth = n
            if eval_type == "min_samples_leaf":
                self.best_model.min_samples_leaf = n
            if eval_type == "min_sample_split":
                self.best_model.min_samples_split = n

        if self.model_type == "knn":
            if self.best_model is None:
                self.best_model = KNeighborsClassifier()
            if eval_type not in self.knn_params:
                raise ValueError("Wrong eval type.")
            if eval_type =="n_neighbors":
                self.best_model.n_neighbors = n
            if eval_type =="p":
                self.best_model.p = n

        if self.model_type == "svm":
            if eval_type =="C":
                self.best_model = SVC(kernel='linear', C=n)
            else:
                raise ValueError("Wrong eval type.")
        if self.model_type == "ada_boost":
            base_tree = tree.DecisionTreeClassifier(
                    random_state=42, max_depth=1)
            if eval_type == "N_estimators":
                self.best_model = AdaBoostClassifier(
                    base_estimator=base_tree, n_estimators=n, learning_rate=1)
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "mlp":
            if self.best_model is None:
                self.best_model = MLPClassifier(
                    alpha=0.0001,
                    solver="adam",
                    max_iter=200,
                    early_stopping=True)
            if eval_type not in self.mlp_params:
                raise ValueError("Wrong eval type.")
            if eval_type == "N_hidden_layers":
                self.best_model.hidden_layer_sizes=n
            if eval_type == "activation_func":
                self.best_model.activation=n

        self.best_model.fit(self.X_train, self.y_train)

        return n

    def save_classification_report(self, dataset):
        y_test_pred = self.best_model.predict(self.X_test)
        report = classification_report(
            self.y_test, y_test_pred, output_dict=True)
        filepath = "table/{}_{}.csv".format(dataset, self.model_type)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(filepath)

    def plot_learning_curves(self, n, scoring_metric_list, dataset):

        if self.best_model == None:
            raise ValueError("Best Model does not exist.")

        if self.model_type == "mlp":
            model_name = self.model_type.title().replace("_", " ")

            fig = plt.figure()
            plt.plot(self.best_model.loss_curve_, 'o-', label='Training Score')
            plt.plot(self.best_model.validation_scores_ , 'o-', label='Validation Score')
            plt.legend(loc='best')
            plt.xlabel('Iterations')
            plt.title("{} - {} {} Curve".format(dataset.upper(), model_name, "Accuracy"))
            # plt.show()

            directory = "plots/{}/".format(dataset)

            filepath = directory + "{}_{}_learning_curve.png".format(self.model_type, "Accuracy")

            fig.savefig(filepath)

        else:
            for scoring_metric in scoring_metric_list:

                train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
                    self.best_model, self.X, self.y, cv=n, scoring=scoring_metric, return_times=True)

                directory = "plots/{}/".format(dataset)

                # Plot the learning curve
                fig = plt.figure()
                plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
                plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Validation Score')
                plt.legend(loc='best')
                plt.xlabel('Number of training samples')
                plt.ylabel(scoring_metric.title())
                model_name = self.model_type.title().replace("_", " ")
                plt.title("{} - {} {} Curve".format(dataset.upper(), model_name, scoring_metric.title()))
                # plt.show()

                filepath = directory + "{}_{}_learning_curve.png".format(self.model_type, scoring_metric)

                fig.savefig(filepath)

            # Plot the fit time as a function of training size.
            fig = plt.figure()
            plt.plot(train_sizes, fit_times.mean(axis=1), 'o-')
            plt.legend(loc='best')
            plt.xlabel('Number of training samples')
            plt.ylabel('Fit time (s)')
            model_name = self.model_type.title().replace("_", " ")
            plt.title("{} - {} Fit Time".format(dataset.upper(), model_name))
            # plt.show()
            filepath = directory + "{}_fit_curve.png".format(self.model_type)

            fig.savefig(filepath)


            # Plot the scoring time as a function of testing size.
            fig = plt.figure()
            plt.plot(train_sizes, score_times.mean(axis=1), 'o-')
            plt.legend(loc='best')
            plt.xlabel('Number of training samples')
            plt.ylabel('Score time (s)')
            model_name = self.model_type.title().replace("_", " ")
            plt.title("{} - {} Score Time".format(dataset.upper(), model_name))
            # plt.show()
            filepath = directory + "{}_score_curve.png".format(self.model_type)
            fig.savefig(filepath)


if __name__ == "__main__":
    dataset_list = [
        "nfl",
        "income",
        ]

    model_list = [
        "decision_tree",
        # "knn", # knn is commented out due to long training times.
        # "svm",
        # "ada_boost",
        # "mlp"
    ]

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

                s = analysis.get_best_model_testing(eval_type=eval_type, eval_metric="AUC")

                print("The optimal parameter for {} is {}".format(eval_type, s))

            n = 10 if model_type == "mlp" else 5

            analysis.plot_learning_curves(
                n=n, scoring_metric_list=['accuracy'], dataset=dataset)
            analysis.save_classification_report(dataset)
