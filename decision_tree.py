
# %%
import os


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from utils import *
import numpy as np

# def eval_overfit(
#     eval_metric: str="rmse",
#     param_iterator: str="depth",
#     eval_range
#     ):
#     metric_test_list, metric_train_list = []
#     for i in eval_range:



if __name__ == "__main__":
    # X, y = get_nfl_data()
    # X, y = get_mushroom_data()
    X, y = get_occupancy_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    nfl_tree = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
    nfl_tree = nfl_tree.fit(X_train, y_train)
    plt.figure(figsize=(16, 16))  # set plot size (denoted in inches)

    tree.plot_tree(
        nfl_tree,
        feature_names=X_train.columns,
        fontsize=10)
    plt.show()

    y_test_pred = nfl_tree.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

    # auc = auc(fpr, tpr)
    # print(auc)

    # Analyze MSE
    mse_test_list = []
    mse_train_list = []

    for i in np.arange(2, 50):
        nfl_tree = tree.DecisionTreeClassifier(random_state=42, max_depth=i)
        nfl_tree = nfl_tree.fit(X_train, y_train)
        y_test_pred = nfl_tree.predict(X_test)
        y_train_pred = nfl_tree.predict(X_train)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test_list.append(mse_test)
        mse_train_list.append(mse_train)
    plt.title("MSE by Max Depth")
    plt.plot(np.arange(2, 50), mse_train_list, label = "Train")
    plt.plot(np.arange(2, 50), mse_test_list, label = "Test")
    plt.legend()
    plt.show()

    # Analyze RUC

    auc_test_list = []
    auc_train_list = []

    for i in np.arange(2, 50):
        nfl_tree = tree.DecisionTreeClassifier(random_state=42, max_depth=i)
        nfl_tree = nfl_tree.fit(X_train, y_train)
        y_test_pred = nfl_tree.predict(X_test)
        y_train_pred = nfl_tree.predict(X_train)

        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
        auc_test = auc(fpr, tpr)

        fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
        auc_train = auc(fpr, tpr)

        auc_test_list.append(auc_test)
        auc_train_list.append(auc_train)

    plt.title("AUC by Max Depth")
    plt.plot(np.arange(2, 50), auc_train_list, label = "Train")
    plt.plot(np.arange(2, 50), auc_test_list, label = "Test")
    plt.legend()
    plt.show()
# %%

    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                  feature_names=iris.feature_names,
    #                  class_names=iris.target_names,
    #                  filled=True, rounded=True,
    #                  special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph




    # y_val = 'TARGET_5Yrs'
    # X_vals = [y_val, 'Name']
    # print(df)
    # drop = True
    # X = df.drop(X_vals, axis=1) if drop else df[X_vals]

    # print(X)
    # df1.drop(['B', 'C'], axis=1)
    # y = df["TARGET_5Yrs"]
    # clf = tree.DecisionTreeClassifier()



