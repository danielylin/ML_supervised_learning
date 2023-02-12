
# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# Use Sklearn for all classification.
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from utils import *
import numpy as np

class ModelAnalysis(object):
    def __init__(self, X, y, model_type):
        self.X = X
        self.y = y

        # Takes in a tuple (metric, eval_type)
        self.train_eval_metrics = {}
        self.test_eval_metrics = {}
        self.eval_range = {}

        self.metrics_list = [
            "MSE", "AUC", "Precision", "Recall", "F1", "Loss", "Accuracy"]

        self.model_type = model_type

    def set_train_test_split(self, test_size=0.3, seed=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=seed
        )

    def _get_model(self, i, eval_type):
        if self.model_type == "decision_tree":
            if eval_type == "Depth":
                model = tree.DecisionTreeClassifier(
                    random_state=42, max_depth=i)
                return model
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "knn":
            if eval_type =="n_neighbors":
                # model = make_pipeline(
                #     MinMaxScaler(),
                #     KNeighborsClassifier(n_neighbors = i))
                model = KNeighborsClassifier(n_neighbors = i)
                return model
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "svm":
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
        elif self.model_type == "MLP":
            if eval_type == "epoch":
                model = MLPClassifier(
                    hidden_layer_sizes=(16,),
                    activation='relu',
                    alpha=0.0001, solver='adam', max_iter=i,
                    random_state=0)
                return model
        else:
            raise ValueError("Model type does not exist.")

    def eval_overfit(self, eval_range, eval_type):
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
        try:
            test = self.test_eval_metrics[eval_type, eval_metric]
            train = self.train_eval_metrics[eval_type, eval_metric]
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
        plt.show()

    def get_best_model_testing(self,
                               eval_type="Depth",
                               eval_metric="AUC",
                               seed=42):

        test = self.test_eval_metrics[eval_type, eval_metric]
        n = self.eval_range[eval_type][np.argmax(test)]

        if self.model_type == "decision_tree":
            if eval_type == "Depth":
                self.best_model = tree.DecisionTreeClassifier(
                    random_state=42, max_depth=n)
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "knn":
            if eval_type =="n_neighbors":
                self.best_model = KNeighborsClassifier(n_neighbors = n)
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "svm":
            if eval_type =="C":
                self.best_model = SVC(kernel='linear', C=n)
            else:
                raise ValueError("Wrong eval type.")
        elif self.model_type == "ada_boost":
            base_tree = tree.DecisionTreeClassifier(
                    random_state=42, max_depth=1)
            if eval_type == "N_estimators":
                self.best_model = AdaBoostClassifier(
                    base_estimator=base_tree, n_estimators=n, learning_rate=1)
            else:
                raise ValueError("Wrong eval type.")

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

    def plot_learning_curves(self, cv_n, scoring_metric_list, dataset):

        if self.best_model == None:
            raise ValueError("Best Model does not exist.")

        for scoring_metric in scoring_metric_list:

            train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
                self.best_model, self.X, self.y, cv=cv_n, scoring=scoring_metric, return_times=True)

            directory = "plots/{}/".format(dataset)

            print(train_scores)
            print(test_scores)

            # Plot the learning curve
            fig = plt.figure()
            plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
            plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Validation Score')
            plt.legend(loc='best')
            plt.xlabel('Number of training samples')
            plt.ylabel(scoring_metric.title())
            model_name = self.model_type.title().replace("_", " ")
            plt.title("{} - {} {} Curve".format(dataset.upper(), model_name, scoring_metric.title()))
            plt.show()

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
        plt.show()
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
        plt.show()
        filepath = directory + "{}_score_curve.png".format(self.model_type)
        fig.savefig(filepath)


if __name__ == "__main__":
    dataset_list = [
        "nfl",
        "income",
        # "nba_rookie",
        # "mushroom",
        # "occupancy",
        # "income"
        ]

    model_list = [
        # "decision_tree",
        # "knn",
        # "svm",
        # "ada_boost",
        "mlp"
    ]

    model_eval_map = {
        "decision_tree":["Depth"],
        "knn": ["n_neighbors"],
        "svm": ["C"],
        "ada_boost": ["N_estimators"]
    }

    model_range_map = {
        "Depth": np.arange(1, 30),
        "n_neighbors": np.arange(1, 500, 5),
        "C": [10**-3,
            10**-2,
            10**-1,
            0.5,
            1
            # 10**2,
            # 10**3
            ],
        "N_estimators": np.arange(30, 305, 5)
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

                n = analysis.get_best_model_testing(eval_type=eval_type)

                print("The optimal parameter for {} is {}".format(eval_type, n))

                analysis.plot_learning_curves(
                    cv_n=5, scoring_metric_list=['accuracy', 'roc_auc'], dataset=dataset)
                analysis.save_classification_report(dataset)






    # iterator(["occupancy", "mushroom", "nfl", "nba_rookie"])

# %%
