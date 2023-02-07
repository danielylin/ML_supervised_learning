# %%
import pandas as pd
pd.options.mode.chained_assignment = None
import glob
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def get_data(dataset:str = "nba"):
    title = dataset.lower()
    if title == "nba":
        df = pd.read_csv("data/nba_logreg.csv")

    if title == "nfl":
        path = os.getcwd()
        pbp_files = glob.glob(os.path.join(path, "data/nfl-pbp/*.csv"))
        df_list = []
        for f in pbp_files:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)

        for data in df_list:
            data.dropna(how = "all", axis=1, inplace=True)

        df = pd.concat(df_list)

    if title == "mushroom":
        df = pd.read_csv("data/agaricus-lepiota.data")

    if title == "occupancy":
        path = os.getcwd()
        pbp_files = glob.glob(os.path.join(path, "data/occupancy_data/*.txt"))
        df_list = []
        for f in pbp_files:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)

            for data in df_list:
                data.dropna(how = "all", axis=1, inplace=True)

        df = pd.concat(df_list)

    return df

def get_nfl_data():
    """This method returns the y and X values for the third down analysis."""
    df = get_data("nfl")
    df_3rd = df[df['Down'] == 3]
    df_3rd = df_3rd[df_3rd["IsNoPlay"] == 0]
    X_vals = ["ToGo", "Formation", "IsPass", "YardLineFixed",
              "Quarter", "Minute", "Second"]
    X = df_3rd[X_vals]
    X.loc[:, "TimeLeftQuarter"] = X["Minute"]*60+X["Second"]
    X = X.drop(["Minute", "Second"], axis = 1)
    ohe = OneHotEncoder(sparse=False)
    cat_features = ["Formation"]
    vals = ohe.fit_transform(df_3rd[cat_features])
    cols = ohe.get_feature_names_out(cat_features)
    converter = lambda x: x.replace(' ', '_')
    cols = list(map(converter, cols))
    categorical_x = pd.DataFrame(vals, columns = cols)
    X = pd.concat([X.drop(cat_features, axis = 1).reset_index(drop=True),
        categorical_x.astype(int).reset_index(drop=True)], axis = 1
        )

    y = df_3rd["SeriesFirstDown"]

    return X, y

def get_nba_data():
    df = get_data("nba")
    y_val = "TARGET_5Yrs"
    X_vals = ["Name", "TARGET_5Yrs"]
    df.dropna(inplace=True)
    X, y = prepare_data(df, y_val, X_vals, drop=True)
    return X, y

def get_mushroom_data():
    df = get_data("mushroom")
    df["p"] = np.where(df["p"] == "p", 1, 0)
    y = df["p"]
    X = df.drop(["p"], axis=1)
    ohe = OneHotEncoder(sparse=False)
    vals = ohe.fit_transform(X)
    cols = ohe.get_feature_names_out(X.columns)
    X = pd.DataFrame(vals, columns = cols)
    return X, y

def get_occupancy_data():
    df = get_data("occupancy")
    y = df["Occupancy"]
    X_cols = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
    X = df[X_cols]
    return X, y

def prepare_data(df, y_val, X_vals, drop=True):
    y = df[y_val]
    X = df.drop(X_vals, axis = 1) if drop else df[X_vals]
    return X, y

if __name__ == "__main__":
    X, y = get_occupancy_data()
    print(X.head())
    print(y)



# %%
