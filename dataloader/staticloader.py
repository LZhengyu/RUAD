import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class StaticLoader():
    def __init__(self):
        # KDDCup 10% Data
        self.kdd_url_data = "dataset/kddcup.data_10_percent.gz"
        # info data (column names, col types)
        self.kdd_url_info = "dataset/kddcup.names"


    def load_kdddata(self):
        # Import info data
        df_info = pd.read_csv(self.kdd_url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
        colnames = df_info.colname.values
        coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
        colnames = np.append(colnames, ["status"])
        coltypes = np.append(coltypes, ["str"])
        # Import data
        df = pd.read_csv(self.kdd_url_data, names=colnames, index_col=False,
                         dtype=dict(zip(colnames, coltypes)))

        # Dumminize
        X = pd.get_dummies(df.iloc[:, :-1]).values

        # Create Traget Flag
        # Anomaly data when status is normal, Otherwise, Not anomaly.
        y = np.where(df.status == "normal.", 1, 0)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=123)
        X_train_normal, y_train_normal = X_train[y_train == 0], y_train[y_train == 0]
        c = int(len(X_train_normal) * 0.05)
        X_train_anomaly, y_train_anomaly = X_train[y_train == 1], y_train[y_train == 1]
        X_train_anomaly, y_train_anomaly = X_train_anomaly[:c], y_train_anomaly[:c]
        X_train = np.concatenate((X_train_normal, X_train_anomaly), axis=0)
        y_train = np.concatenate((y_train_normal, y_train_anomaly), axis=0)

        return X_train,y_train,X_test,y_test
