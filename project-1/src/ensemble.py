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


if __name__ == "__main__":
    from model_factory import ModelFactory
    import torch
    from sklearn.metrics import f1_score, accuracy_score

    model_fac = ModelFactory()
    model_names = ["CnnWithResidualConnection", "CnnEncoder"]
    # list of model loaders :
    # getting class from factory and creating an instance
    lazy_models = [lambda : model_fac.get(name)() for name in model_names]
    ensemble = ModelEnsembler(config={"lazy_loading": True})
    ensemble.add_models(lazy_models, [1, 2])
    from data_loader import DataLoaderUtil, DATA_MITBIH
    _, _, test_loader = DataLoaderUtil().get_data_loaders(
        dataset_name=DATA_MITBIH,
        test_batch_size=100
    )
    for idx, batch_data in enumerate(test_loader):
        x_true, y_true = batch_data
        y_pred_prob = ensemble.predict(x_true)
        y_pred = torch.argmax(y_pred_prob, dim=1)
        print(f"Acc: {accuracy_score(y_true, y_pred)}")
        if idx == 1:
            break

    def load_model():
        path_ = "runs/2022-03-23_225626__CnnWithResidualConnection/best_model.ckpt"
        model_weights = torch.load(path_)
        model = model_fac.get("CnnWithResidualConnection")()
        model.load_state_dict(model_weights)
        return model
    lazy_models = [load_model, load_model]
    ensemble = ModelEnsembler(config={"lazy_loading": True})
    ensemble.add_models(lazy_models, [1, 2])
    for idx, batch_data in enumerate(test_loader):
        x_true, y_true = batch_data
        y_pred_prob = ensemble.predict(x_true)
        y_pred = torch.argmax(y_pred_prob, dim=1)
        print(f"Acc: {accuracy_score(y_true, y_pred)}")
        if idx == 1:
            break
    
    lazy_models = [load_model]
    ensemble = ModelEnsembler(config={"lazy_loading": True})
    ensemble.add_models(lazy_models, [2])
    for idx, batch_data in enumerate(test_loader):
        x_true, y_true = batch_data
        y_pred_prob = ensemble.predict(x_true)
        y_pred = torch.argmax(y_pred_prob, dim=1)
        print(f"Acc: {accuracy_score(y_true, y_pred)}")
        if idx == 1:
            break
         




            

