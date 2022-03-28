from svm import load_params, dataloader_d, class_num
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from data_loader import MITBIH, PTBDB, DataLoaderUtil
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from model_factory import TrainedModelFactory, RnnModelMITBIH, CnnWithResidualConnection

class BaseEnsembler:
    def __init__(self, *args, **kwargs) -> None:
        self.models = []
        self.sampling_count = []

    def add_models(self, model_list, sample_counts=None):
        if sample_counts is None:
            sample_counts = [1]*len(model_list)
        for idx in range(len(model_list)):
            self.add_model(model_list[idx], sample_counts[idx])

    def add_model(self, model, sample_count=1):
        self.models.append(model)
        self.sampling_count.append(sample_count)
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def aggregate(self, *args, **kwargs):
        raise NotImplementedError()


class ModelEnsembler(BaseEnsembler):
    def __init__(self, config=None) -> None:
        super().__init__()
        if config is None:
            config = {"lazy_loading": False}
        self.config = config

        self.lazy_loading = self.config["lazy_loading"]

    def predict(self, x):
        """
        """
        aggregate_prediction_prob = None
        model_count = len(self.models)
        for idx, model in enumerate(self.models):
            if self.lazy_loading:
                model = model() # in this case model should be a callable
                # which loads the model and returns loaded model 
            num_sample = self.sampling_count[idx]
            for i in range(num_sample):
                if hasattr(model, "predict"):
                    delta = model.predict(x)/num_sample
                else:
                    model.eval()
                    delta = model(x)
                if aggregate_prediction_prob is None:
                    aggregate_prediction_prob = delta
                else:
                    aggregate_prediction_prob = aggregate_prediction_prob \
                        + delta
            if self.lazy_loading:
                del model
        aggregate_prediction_prob = aggregate_prediction_prob/model_count
        return aggregate_prediction_prob

def train_sklearn_ensemble(dataset, n_estimators, max_samples, n_jobs, model_path=None):
    x_train, y_train, x_test, y_test = dataloader_d[dataset]().load_data()
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    model_params = load_params(model_path)
    model_params["random_state"] = None
    model_params["probability"] = True
    model = SVC(**model_params)
    ensemble = BaggingClassifier(base_estimator=model, n_estimators=n_estimators,
        max_samples=max_samples, n_jobs=n_jobs, verbose=2)
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    print('Training ensemble')
    ensemble.fit(x_train, y_train)
    y_pred = ensemble.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("Test f1 score : %s "% f1)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy score : %s "% acc)

def test():
    from model_factory import ModelFactory
    import torch
    from sklearn.metrics import f1_score, accuracy_score

    # list of model loaders :
    # getting class from factory and creating an instance
    def load_model(model_class):
        lazy_model_loader = TrainedModelFactory().get_lazy_loader(model_class)
        #model = lazy_model_loader()
        return lazy_model_loader

    ensemble = ModelEnsembler(config={"lazy_loading": True})
    #models = [load_model("RnnModelMITBIH")]
    models = [load_model("CnnWithResidualConnection"), load_model("RnnModelMITBIH")]
    ensemble.add_models(models)

    from data_loader import DataLoaderUtil, DATA_MITBIH, DATA_PTBDB
    _, _, test_loader = DataLoaderUtil().get_data_loaders(
        dataset_name=DATA_MITBIH,
        test_batch_size=100
    )

    g_truth = []
    preds = []
    for idx, batch_data in enumerate(test_loader):
        x_true, y_true = batch_data
        y_pred_prob = ensemble.predict(x_true)
        batch_pred = torch.argmax(y_pred_prob, dim=1)
        g_truth.append(y_true)
        preds.append(batch_pred)
    y_pred = torch.concat(preds)
    y_true = torch.concat(g_truth)
    print(f"Acc: {accuracy_score(y_true, y_pred)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--svm', action='store_true')
    parser.add_argument('-samples', type=int, default=None)
    parser.add_argument('-models', type=int, default=2)
    parser.add_argument('-jobs', type=int, default=1)
    parser.add_argument('-modelpath', default=None)
    def_samples = {MITBIH: 40000, PTBDB: 11641}
    args = parser.parse_args()
    if args.svm:
        train_sklearn_ensemble(
            args.dataset,
            args.models,
            args.samples if args.samples is not None else def_samples[args.dataset],
            args.jobs, model_path=args.modelpath
        )
    else:
        test()