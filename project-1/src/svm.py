import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score
from data_loader import MITBIH, MITBIHDataLoader, PTBDataLoader, DataLoaderUtil, MITBIH, PTBDB
from util import get_timestamp_str
import pickle
import numpy as np
from argparse import ArgumentParser
import time

dataloader_d = {
    MITBIH: MITBIHDataLoader,
    PTBDB: PTBDataLoader
}

class_num = {
    MITBIH: 5,
    PTBDB: 2
}

class_subsample = {
    MITBIH: 1,
    PTBDB: 0
}

def sample_max_class_(x, y, class_idx, counts, max_n):
    outx = []
    outy = []
    for i, class_i in enumerate(class_idx):
        if max_n < counts[i]:
            x_sample, y_sample = resample(x[class_i], y[class_i], replace=False, n_samples=max_n, random_state=0)
            outx.append(x_sample)
            outy.append(y_sample)
        else:
            outx.append(x[class_i])
            outy.append(y[class_i])
    return np.concatenate(outx), np.concatenate(outy)

def min_sample(x, y, class_n, k=1):
    idx = [(y == i) for i in range(class_n)]
    counts = list(map(np.sum, idx))
    s = sorted(counts)
    max_n = s[-k-1]
    return sample_max_class_(x, y, idx, counts, max_n)
    
class SVCSubsample(SVC):
    def __init__(self, class_n=None, k=None, **kwargs):
        super().__init__(**kwargs)
        self.class_n = class_n
        self.k = k

    def fit(self, X, y, *args, **kwargs):
        if self.class_n == None:
            self.class_n = np.max(y)+1 # assumes classes from 0 to N-1
        if self.k != None:
            x_sample, y_sample = min_sample(X, y, self.class_n, self.k)
        else:
            x_sample, y_sample = X, y
        super().fit(x_sample, y_sample, *args, **kwargs)

def run(dataset, iters, jobs):
    print(f'Running {iters} iterations on {jobs} jobs. Dataset {dataset}')
    dataloader = dataloader_d[dataset]()
    x_train, y_train, x_test, y_test = dataloader.load_data()
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    class_n = class_num[dataset]
    subsample = class_subsample[dataset]
    x_sample, y_sample = min_sample(x_train, y_train, class_n, subsample)
    scaler = StandardScaler()
    scaler.fit(x_sample)
    x_sample = scaler.transform(x_sample)
    x_test = scaler.transform(x_test)
    model = SVC()
    param_grid = {
        "C": np.logspace(-3,1),
        "kernel": ['poly', 'rbf', 'sigmoid'],
        "degree": np.linspace(3,6,4),
        "gamma": np.logspace(-3,-0.5)
    }
    search = RandomizedSearchCV(model, param_grid, n_iter=iters, scoring=['f1_macro','accuracy','balanced_accuracy'], refit="f1_macro", n_jobs=jobs, verbose=2)
    trained_model = search.fit(x_sample, y_sample)
    out_path_stats = "svm_%s_stats.csv" % get_timestamp_str()
    out_path_model = "svm_%s_model.pickle" % get_timestamp_str()
    pd.DataFrame(search.cv_results_).to_csv(out_path_stats)
    pickle.dump(trained_model, open(out_path_model, 'bw'))
    y_pred = trained_model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("Test f1 score : %s "% f1)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy score : %s "% acc)

def test_sample_rates(dataset, iters, model_path):
    model_params = None
    if model_path == None:
        print('Training default SVC')
        model_params = {}
    else:
        print(f'Training with params from {model_path}')
        model_example = pickle.load(open(model_path, 'rb'))
        model_params = model_example.best_estimator_.get_params()
        print('Params', model_params)
    model_params['random_state'] = 0
    dataloader = dataloader_d[dataset]()
    x_train, y_train, x_test, y_test = dataloader.load_data()
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    class_n = class_num[dataset]
    idx = [(y_train == i) for i in range(class_n)]
    counts = list(map(np.sum, idx))
    start = min(counts)
    stop = max(counts)

    f1_log = []
    acc_log = []
    time_log = []

    for n in np.linspace(start, stop, iters):
        n = int(n)
        x_sample, y_sample = sample_max_class_(x_train, y_train, idx, counts, n)
        model = SVC(**model_params)
        start_time = time.time()
        model.fit(x_sample, y_sample)
        total_time = (time.time() - start_time)
        y_pred = model.predict(x_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        f1_log.append(f1)
        acc_log.append(acc)
        time_log.append(total_time)
        print(f'Trained with max {n}. f1 {f1}. Accuracy {acc}. Time {total_time}s')

    data = pd.DataFrame({
        'n': map(int,np.linspace(start, stop, iters)),
        'f1_test': f1_log,
        'acc_test': acc_log,
        'time_train': time_log
    })
    data.to_csv("sample_svm_%s_stats.csv" % get_timestamp_str())

if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='action',dest='action')
    train_subp = subparsers.add_parser('train')
    train_subp.add_argument('dataset', choices=[MITBIH, PTBDB])
    train_subp.add_argument('-iter', action='store', type=int, default=10)
    train_subp.add_argument('-jobs', action='store', type=int, default=1)
    samples_subp = subparsers.add_parser('samples')
    samples_subp.add_argument('dataset', choices=[MITBIH, PTBDB])
    samples_subp.add_argument('-model', action='store', default=None)
    samples_subp.add_argument('-iter', action='store', type=int, default=10)
    
    args = parser.parse_args()
    if args.action == 'train':
        run(args.dataset, args.iter, args.jobs)
    elif args.action == 'samples':
        test_sample_rates(args.dataset, args.iter, args.model)