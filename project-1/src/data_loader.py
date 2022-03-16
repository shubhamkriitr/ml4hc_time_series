# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


DATASET_LOC_MITBIH_TRAIN = "../resources/input/mitbih_train.csv"
DATASET_LOC_MITBIH_TEST = "../resources/input/mitbih_test.csv"
DATASET_LOC_PTBDB_NORMAL = "../resources/input/ptbdb_normal.csv"
DATASET_LOC_PTBDB_ABNORMAL = "../resources/input/ptbdb_abnormal.csv"
DATA_MITBIH = "Dataset 1"
DATA_PTBDB = "Dataset 2"
class ClassificationDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DataLoaderUtil:
    def __init__(self) -> None:
        self.config
    def prepare_data_loader(self, dataset_name):
        if dataset_name == DATA_MITBIH:
            dataloader = MITBIHDataLoader()
        elif dataset_name == DATA_PTBDB:
            dataloader = PTBDataLoader()

        return dataloader
        
    def load_data(self, dataset_name):
        dataloader = self.prepare_data_loader(dataset_name)
        x_train, y_train, x_test, y_test = dataloader.load_data()
        return x_train, y_train, x_test, y_test
    
    def get_datasets_split(self, dataset_name, val_split=0.1):
        x_train, y_train, x_test, y_test = self.load_data(dataset_name)

        #train
        train_dataset = ClassificationDataset(x_train, y_train)
        test_dataset = ClassificationDataset(x_test, y_test)
        val_dataset = None # TODO: add validation splitting

        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self,  dataset_name, train_batch_size=8, 
        val_batch_size=8, test_batch_size=8, train_shuffle=True, val_split=0.1):
        train_dataset, val_dataset, test_dataset = self.get_datasets_split(
            dataset_name, val_split
        )
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                shuffle=train_shuffle)
        
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
        
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
        
        return train_loader, val_loader, test_loader

        
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
        df_train = pd.read_csv(str(Path(DATASET_LOC_MITBIH_TRAIN)), header=None)
        df_train = df_train.sample(frac=1)
        df_test = pd.read_csv(str(Path(DATASET_LOC_MITBIH_TEST)), header=None)

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
        df_1 = pd.read_csv(str(Path(DATASET_LOC_PTBDB_NORMAL)), header=None)
        df_2 = pd.read_csv(str(Path(DATASET_LOC_PTBDB_ABNORMAL)), header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
        return X, Y, X_test, Y_test

