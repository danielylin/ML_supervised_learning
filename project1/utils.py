# %%
import pandas as pd
pd.options.mode.chained_assignment = None
import glob
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import numpy as np

def get_data(dataset:str = "nba_rookie"):
    title = dataset.lower()

    if title not in ["nba_rookie", "nfl", "mushroom", "occupancy", "income"]:
        raise ValueError("Dataset {} does not exist.".format(title))

    if title == "nba_rookie":
        df = pd.read_csv("data/nba_logreg.csv")
        X, y = prep_nba_rookie_data(df)

    if title == "nfl":
        path = os.getcwd()
        pbp_files = glob.glob(os.path.join(path, "data/nfl-pbp/*.csv"))
        df_list = []
        for f in pbp_files:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)

        for data in df_list:
            data.dropna(how = "all", axis=1, inplace=True)

        df = pd.concat(df_list, ignore_index=True)
        X, y = prep_nfl_data(df)

    if title == "mushroom":
        colnames = ['class',
                    'cap-shape',
                    'cap-surface',
                    'cap-color',
                    'bruises',
                    'odor',
                    'gill-attachment',
                    'gill-spacing',
                    'gill-size',
                    'gill-color',
                    'stalk-shape',
                    'stalk-root',
                    'stalk-surface-above-ring',
                    'stalk-surface-below-ring',
                    'stalk-color-above-ring',
                    'stalk-color-below-ring',
                    'veil-type',
                    'veil-color',
                    'ring-number',
                    'ring-type',
                    'spore-print-color',
                    'population',
                    'habitat']
        df = pd.read_csv("data/agaricus-lepiota.data", header=None, names=colnames)
        X, y = prep_mushroom_data(df)


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
        X, y = prep_occupancy_data(df)

    if title == "income":
        df = pd.read_csv("data/adult.csv")
        X, y = prep_income_data(df)

    return X, y

def prep_nfl_data(df):
    """This method returns the y and X values for the third down analysis."""
    # df = get_data("nfl")
    df_3rd = df[df['Down'] == 3]
    df_3rd = df_3rd[df_3rd["IsNoPlay"] == 0]
    X_vals = ["ToGo", "Formation", "IsPass", "YardLineFixed",
              "Quarter", "Minute", "Second", "PassType", "RushDirection",
              "OffenseTeam", "DefenseTeam"]
    X = df_3rd[X_vals]
    X.loc[:, "TimeLeftQuarter"] = X["Minute"]*60+X["Second"]
    X = X.drop(["Minute", "Second"], axis = 1)
    ohe = OneHotEncoder(sparse=False)

    # encode StandardScaler
    # if scale==True:
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(exclude=['object'])
    numerical_vals = scaler.fit_transform(numerical_features)

    X[numerical_features.columns] = numerical_vals

    cat_features = ["Formation", "RushDirection", "PassType", "OffenseTeam", "DefenseTeam"]
    vals = ohe.fit_transform(df_3rd[cat_features])
    cols = ohe.get_feature_names_out(cat_features)
    converter = lambda x: x.replace(' ', '_')
    cols = list(map(converter, cols))
    categorical_x = pd.DataFrame(vals, columns = cols)
    categorical_x.drop(
        columns=[col for col in categorical_x.columns if "nan" in col],
        inplace=True)

    X = pd.concat([X.drop(cat_features, axis = 1).reset_index(drop=True),
        categorical_x.astype(int).reset_index(drop=True)], axis = 1
        )

    y = df_3rd["SeriesFirstDown"]
    return X, y

def prep_nba_rookie_data(df):
    y_val = "TARGET_5Yrs"
    X_vals = ["Name", "TARGET_5Yrs"]
    df.dropna(inplace=True)
    X, y = prep_data(df, y_val, X_vals, drop=True)
    return X, y

def prep_mushroom_data(df):
    df["class"] = np.where(df["class"] == "p", 1, 0)
    y = df["class"]
    X = df.drop(["class"], axis=1)
    ohe = OneHotEncoder(sparse=False)
    vals = ohe.fit_transform(X)
    cols = ohe.get_feature_names_out(X.columns)
    X = pd.DataFrame(vals, columns = cols)
    return X, y

def prep_occupancy_data(df):
    df = get_data("occupancy")
    y = df["Occupancy"]
    X_cols = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
    X = df[X_cols]
    return X, y

def prep_income_data(df):
    df = df.replace("?", "unknown")
    y = np.where(df["income"] == ">50K", 1, 0)
    X_cols = [
        "age",
        "workclass",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"]

    X = df[X_cols]

    scaler = StandardScaler()
    numerical_features = X.select_dtypes(exclude=['object'])
    numerical_vals = scaler.fit_transform(numerical_features)

    X[numerical_features.columns] = numerical_vals

    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country"]

    ohe = OneHotEncoder(sparse=False)

    vals = ohe.fit_transform(df[cat_features])
    cols = ohe.get_feature_names_out(cat_features)
    converter = lambda x: x.replace(' ', '_')
    cols = list(map(converter, cols))
    categorical_x = pd.DataFrame(vals, columns = cols)
    X = pd.concat([X.drop(cat_features, axis = 1).reset_index(drop=True),
        categorical_x.astype(int).reset_index(drop=True)], axis = 1
        )
    return X, y

def prep_data(df, y_val, X_vals, drop=True):
    y = df[y_val]
    X = df.drop(X_vals, axis = 1) if drop else df[X_vals]
    return X, y


if __name__ == "__main__":
    X, y = get_data("nfl")
    print(X)
    print(y)

    print(sum(y)/len(y))


# %%