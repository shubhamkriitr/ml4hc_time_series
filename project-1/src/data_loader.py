# Data loader classes for arrythmia and PTB datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from util import PROJECTPATH
import torch

MITBIH = "mitbih"
PTBDB = "ptbdb"

DATASET_LOC_MITBIH_TRAIN = str(Path(PROJECTPATH)/"input/mitbih_train.csv")
DATASET_LOC_MITBIH_TEST = str(Path(PROJECTPATH)/"input/mitbih_test.csv")
DATASET_LOC_PTBDB_NORMAL = str(Path(PROJECTPATH)/"input/ptbdb_normal.csv")
DATASET_LOC_PTBDB_ABNORMAL = str(Path(PROJECTPATH)/"input/ptbdb_abnormal.csv")

# these class names are used in config and in DataLoaderUtil
# to refer the the two datsets for different tasks
DATA_MITBIH = "MITBIHDataLoader"
DATA_PTBDB = "PTBDataLoader"
DATA_MITBIH_BAL = "BalancedMITBIHDataLoader"
DATA_MITBIH_AUTO_ENC = "MITBIHDataLoaderForAutoEncoder" # for training auto encoder
DATA_MITBIH_AUTO_ENC_BAL = "BalancedMITBIHDataLoaderForAutoEncoder"
DATA_PTBDB_AUTO_ENC = "PTBDataLoaderForAutoEncoder"


MAX_SIZE_IN_BALANCED_DATASET = 3000 # max num of samples


class TaskDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# It is functionally same as TaskDataset
# but creating this class anyways for readability.
# Note: y will be same as x in this case
class AutoEncoderDataset(TaskDataset):
    def __init__(self, x, y):
        super().__init__(x, y)


class DataLoaderUtil:
    def __init__(self) -> None:
        self.x_data_type = torch.float32
        self.y_data_type = None # set in prepare_data_loader

    @staticmethod
    def print_class_percentages(y):
        classes, frequencies = torch.unique(y, return_counts=True)
        percentages = 100*frequencies/torch.sum(frequencies)
        print(classes, frequencies)
        print(percentages)

    def get_split(self, x, y, val_ratio):
        data = torch.cat([x, torch.unsqueeze(y, dim=1).unsqueeze(dim=2)], dim=2)
        train_data, val_data = train_test_split(data, test_size=val_ratio, random_state=1337, stratify=data[:, :, -1]) # last col of last dim
        
        return (train_data[:, :, 0:-1].type(self.x_data_type),
         train_data[:, 0, -1].type(self.y_data_type),
          val_data[:, :, 0:-1].type(self.x_data_type),
           val_data[:, 0, -1].type(self.y_data_type) )

    @staticmethod
    def sanity_check(x_orig, y_orig, x_chunk, y_chunk):
         # warning: not optimized
        for i in range(x_chunk.shape[0]):
            for j in range(x_orig.shape[0]):
                if torch.all(x_orig[j] == x_chunk[i]):
                    assert y_orig[j] == y_chunk[i]
                else:
                    pass
        

    def prepare_data_loader(self, dataset_name):
        self.y_data_type = torch.long
        if dataset_name == DATA_MITBIH:
            dataloader = MITBIHDataLoader()
        elif dataset_name == DATA_MITBIH_BAL:
            dataloader = BalancedMITBIHDataLoader()
        elif dataset_name == DATA_PTBDB:
            self.y_data_type = torch.float32
            dataloader = PTBDataLoader()
        elif dataset_name == DATA_MITBIH_AUTO_ENC:
            self.y_data_type = torch.float32
            dataloader = MITBIHDataLoaderForAutoEncoder()
        elif dataset_name == DATA_MITBIH_AUTO_ENC_BAL:
            self.y_data_type = torch.float32
            dataloader = BalancedMITBIHDataLoaderForAutoEncoder()
        elif dataset_name == DATA_PTBDB_AUTO_ENC:
            self.y_data_type = torch.float32
            dataloader = PTBDataLoaderForAutoEncoder()

        return dataloader
        
    def load_data(self, dataset_name):
        dataloader = self.prepare_data_loader(dataset_name)
        x_train, y_train, x_test, y_test = dataloader.load_data()

        # get torch tensors
        x_train, x_test = [torch.tensor(data=data_item, dtype=self.x_data_type) 
                            for data_item in [x_train, x_test]]
        y_train, y_test = [torch.tensor(data=data_item, dtype=self.y_data_type) 
                             for data_item in [y_train, y_test]]

        # Pytorch expects channel first dimension ordering
        # therefore transposing to bring channel first
        x_train = torch.transpose(x_train, 1, 2)
        x_test = torch.transpose(x_test, 1, 2)

        if dataset_name in [DATA_MITBIH_AUTO_ENC, DATA_PTBDB_AUTO_ENC,
        DATA_MITBIH_AUTO_ENC_BAL]:
            # ouput for encoder-decoder 
            y_train = x_train
            y_test = x_test

        return x_train, y_train, x_test, y_test
    
    def get_datasets_split(self, dataset_name, val_split=None):
        x_train, y_train, x_test, y_test = self.load_data(dataset_name)
        train_indices, val_indices = None, None
        val_dataset = None # TODO: add validation splitting

        if val_split is not None:
            x_train, y_train, x_val, y_val = self.get_split(
                x_train, y_train, val_split
            )
            
            val_dataset = TaskDataset(x_val, y_val)
            print("Validation:")
            self.print_class_percentages(y_val)

            
        #train
        train_dataset = TaskDataset(x_train, y_train)
        test_dataset = TaskDataset(x_test, y_test)
        
        print("Training:")
        self.print_class_percentages(y_train)
        
        print("Testing:")
        self.print_class_percentages(y_test)

        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self,  dataset_name, train_batch_size=8, 
        val_batch_size=8, test_batch_size=8, train_shuffle=True, val_split=None
        ):
        train_dataset, val_dataset, test_dataset = self.get_datasets_split(
            dataset_name, val_split
        )
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                shuffle=train_shuffle)
        
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
        
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
        
        return train_loader, val_loader, test_loader

class DataLoaderUtilMini(DataLoaderUtil):
    def load_data(self, dataset_name):
        x_train, y_train, x_test, y_test =  super().load_data(dataset_name)
        x_train, y_train, x_test, y_test = [data_item[0:100] for data_item in
            [x_train, y_train, x_test, y_test]]
        return x_train, y_train, x_test, y_test
        
class MITBIHDataLoader:
    def __init__(self):
        pass

    def load_data(self):
        """
        Assumes data is in input/
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

class BalancedMITBIHDataLoader(MITBIHDataLoader):
    def load_data(self):
        x, y, x_test, y_test =  super().load_data()
        max_size = 3000
        x, y = balance_dataset(max_size, x, y)

        return x, y, x_test, y_test

def balance_dataset(max_size, x, y):
    labels, frequencies = np.unique(y, return_counts=True)
    
    final_selection_indices = []
    for idx in range(labels.shape[0]):
        current_label = labels[idx]
        chunk_indices = np.where(y == current_label)[0]

        if chunk_indices.shape[0] > max_size:
            chunk_indices = chunk_indices[0:max_size]
        
        final_selection_indices.append(chunk_indices)

    final_selection_indices = np.concatenate(final_selection_indices, axis=0)    
    x = x[final_selection_indices]
    y = y[final_selection_indices]
    return x, y


class PTBDataLoader:
    def __init__(self):
        pass

    def load_data(self):
        """
        Assumes data is in input/
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


class MITBIHDataLoaderForAutoEncoder(MITBIHDataLoader):
    def __init__(self):
        super().__init__()
    
    def load_data(self):
        x, _, x_test, _ =  super().load_data()
        y = x
        y_test = x_test
        return x, y, x_test, y_test

class BalancedMITBIHDataLoaderForAutoEncoder(MITBIHDataLoader):
    def load_data(self):
        x, true_labels, x_test, _ =  super().load_data()
        y_test = x_test

        x, true_labels = balance_dataset(MAX_SIZE_IN_BALANCED_DATASET,
                            x, true_labels)
        y = x
        return x, y, x_test, y_test

class PTBDataLoaderForAutoEncoder(PTBDataLoader):
    def __init__(self):
        super().__init__()
    
    def load_data(self):
        x, _, x_test, _ =  super().load_data()
        y = x
        y_test = x_test
        return x, y, x_test, y_test

if __name__ == "__main__":
    dataloader_util = DataLoaderUtil()
    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(DATA_MITBIH)
    for batch_data in train_loader:
        x, y = batch_data
        z = x