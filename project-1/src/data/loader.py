# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

class MITBIHDataLoader:
    def __init__(self):
        pass

    def load_data(self):
        """
        Assumes data is in resources/input
        Returns:
        X:      np array (87554, 187, 1)
        Y:      np array (87554,)
        X_test: np array (21892, 187, 1)
        Y_test: np_array (21892,)
        """
        df_train = pd.read_csv(str(Path("../../resources/input/mitbih_train.csv")), header=None)
        df_train = df_train.sample(frac=1)
        df_test = pd.read_csv(str(Path("../../resources/input/mitbih_test.csv")), header=None)

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
        return X, Y, X_test, Y_test

class PTBDataLoader:
    def __init__(self):
        pass

    def load_data(self):
        """
        Assumes data is in resources/input
        Returns:
        X:      np array (11641, 187, 1)
        Y:      np array (11641,)
        X_test: np array (2911, 187, 1)
        Y_test: np_array (2911,)
        """
        df_1 = pd.read_csv(str(Path("../../resources/input/ptbdb_normal.csv")), header=None)
        df_2 = pd.read_csv(str(Path("../../resources/input/ptbdb_abnormal.csv")), header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
        return X, Y, X_test, Y_test