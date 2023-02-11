
# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from utils import *
import numpy as np

class DecisionTreeAnalysis(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.train_mse_list = []
        self.test_mse_list = []
        self.train_auc_list = []
        self.test_auc_list = []
        self.train_precision_list = []
        self.test_precision_list = []
        self.train_recall_list = []
        self.test_recall_list = []
        self.train_f1_list = []
        self.test_f1_list = []

        self.eval_range = {}
        self.metrics_map = {
            "MSE": (self.test_mse_list, self.train_mse_list),
            "AUC": (self.test_auc_list, self.train_auc_list),
            "Precision": (self.test_precision_list, self.train_precision_list),
            "Recall": (self.test_recall_list, self.train_recall_list),
            "F1": (self.test_f1_list, self.train_f1_list)
        }

    def set_train_test_split(self, test_size=0.3, seed=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=seed
        )


    def eval_overfit(self, eval_range, eval_type):
        for i in eval_range:
            if eval_type == "Depth":
                dt = tree.DecisionTreeClassifier(random_state=42, max_depth=i)
                dt = dt.fit(self.X_train, self.y_train)

                y_test_pred = dt.predict(self.X_test)
                y_train_pred = dt.predict(self.X_train)

                # Calculate MSE though it is not a great metric for classification.
                mse_test = mean_squared_error(self.y_test, y_test_pred)
                mse_train = mean_squared_error(self.y_train, y_train_pred)
                self.train_mse_list.append(mse_train)
                self.test_mse_list.append(mse_test)

                # Calculate fpr, tpr to generate AUC.
                # AUC is the measure of FPR to TPR at various thresholds.
                fpr, tpr, thresholds = roc_curve(self.y_train, y_train_pred)
                auc_train = auc(fpr, tpr)
                fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred)
                auc_test = auc(fpr, tpr)

                self.train_auc_list.append(auc_train)
                self.test_auc_list.append(auc_test)

                # Precision score.
                precision_train = precision_score(self.y_train, y_train_pred)
                precision_test = precision_score(self.y_test, y_test_pred)

                self.train_precision_list.append(precision_train)
                self.test_precision_list.append(precision_test)

                # Recall score.
                recall_train = recall_score(self.y_train, y_train_pred)
                recall_test = recall_score(self.y_test, y_test_pred)

                self.train_recall_list.append(recall_train)
                self.test_recall_list.append(recall_test)

                # F1 score
                f1_train = f1_score(self.y_train, y_train_pred)
                f1_test = f1_score(self.y_test, y_test_pred)
                self.train_f1_list.append(f1_train)
                self.test_f1_list.append(f1_test)

        self.eval_range[eval_type] = eval_range

    def plot_overfit(self, eval_metric, eval_type, filepath):

        try:
            test, train = self.metrics_map[eval_metric]
            eval_range = self.eval_range[eval_type]
        except KeyError:
            raise ValueError("Wrong eval metric")


        fig = plt.figure()
        plt.title("{} by {}".format(eval_metric, eval_type))
        plt.plot(eval_range, train, label="Train")
        plt.plot(eval_range, test, label="Test")
        plt.legend()
        plt.xlabel("{}".format(eval_type))
        plt.ylabel(eval_metric)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(filepath, dpi=300)
        plt.show()

    def get_best_model_testing(self,
                               eval_metric="AUC",
                               eval_type="Depth",
                               seed=42):
        test, __ = self.metrics_map[eval_metric]
        n = self.eval_range[eval_type][np.argmax(test)]
        self.best_model= tree.DecisionTreeClassifier(
            random_state=42, max_depth=n)

        self.best_model.fit(self.X_train, self.y_train)

        return n

    def plot_best_model(self):
        plt.figure(figsize=(16, 16))
        tree.plot_tree(
            self.best_model,
            feature_names=self.X_train.columns,
            fontsize=10)
        plt.show()

def run_tree_analysis(data_name):
    if data_name == "occupancy":
        X, y = get_occupancy_data()
    elif data_name == "mushroom":
        X, y = get_mushroom_data()
    elif data_name == "nfl":
        X, y = get_nfl_data()
    elif data_name == "nba":
        X, y = get_nba_data()
    else:
        ValueError("Data does not exist.")

    dt = DecisionTreeAnalysis(X, y)
    dt.set_train_test_split()

    eval_range = np.arange(2, 50)
    eval_type = "Depth"
    dt.eval_overfit(eval_range, eval_type)
    eval_metrics = ["MSE", "AUC", "Precision", "Recall", "F1"]

    for metric in eval_metrics:
        dt.plot_overfit(
            metric, eval_type, "plots/{}/dt_{}_{}.png".format(
                data_name, metric.lower(), eval_type.lower()))

    dt.get_best_model_testing()
    dt.plot_best_model()

def iterator(data_names):
    for data_name in data_names:
        run_tree_analysis(data_name)

if __name__ == "__main__":
    iterator(["occupancy", "mushroom", "nfl", "nba"])

# %%
