import torch
from torch.nn import functional as F
from torchdrug import core, layers, metrics, tasks
from torchdrug.layers import functional
from sklearn.metrics import matthews_corrcoef as mcc_sklearn


class NodePropertyPrediction(tasks.Task, core.Configurable):
    ## Nov 4: MinGyu Choi
    ### I found out that some problems emerge if we directly use NodePropertyPrediction.
    ### I guess this is the problem of target setting, but correcting them seems to require a lot of time...
    ### Therefore I just edited some tensor shapes for this specific case.
    ### It might be cumbersome to do; but please understand.

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("macro_auprc", "macro_auroc"), num_mlp_layer=1,
                 normalization=True, num_class=None, verbose=0,
                 graph_construction_model=None,
                 threshold=-1.5,
                 bce_weight=1.0,
                 use_rus=False,
                 rus_seed=0,
                 undersample_rate=0.1,
                 ):
        super(NodePropertyPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.graph_construction_model = graph_construction_model
        self.threshold = threshold
        self.bce_weight = bce_weight
        self.use_rus = use_rus
        self.rus_seed = rus_seed
        self.undersample_rate = undersample_rate

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation on the training set.
        """
        self.view = getattr(train_set[0]["graph"], "view", "atom")
        values = torch.cat([data["graph"].target for data in train_set])
        mean = values.float().mean()
        std = values.float().std()
        if values.dtype == torch.long:
            num_class = values.max().item()
            if num_class > 1 or "bce" not in self.criterion:
                num_class += 1
        else:
            num_class = 1

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [self.num_class])

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        node_feature = graph.node_feature.float()
        # if node_feature is of type `torch.sparse.FloatTensor`
        # Error like "Could not run 'aten::view' with arguments from the 'SparseCPU' backend" occurs. Thus do this conversion.
        if 'sparse' in node_feature.type():
            node_feature = node_feature.to_dense()
        output = self.model(graph, node_feature, all_loss=all_loss, metric=metric)
        if self.view in ["node", "atom"]:
            output_feature = output["node_feature"]
        else:
            output_feature = output.get("residue_feature", output.get("node_feature"))
        pred = self.mlp(output_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        size = batch["graph"].num_nodes if self.view in ["node", "atom"] else batch["graph"].num_residues
        return {
            "label": batch["graph"].target,
            "mask": batch["graph"].mask,
            "size": size
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device).view(1)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        labeled = ~torch.isnan(target["label"]) & target["mask"]

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"].float(), reduction="none", pos_weight=self.bce_weight)
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target["label"], reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            
            # RUS should only be applied to training set, since calculation of valid_bce depends on this method and it should not be affected by RUS.
            if self.use_rus and self.training:
                torch.random.manual_seed(self.rus_seed)
                random_tensor = torch.rand(loss.shape, device=self.device)
                cost_mask = ((target['label'] == 1) + (random_tensor < self.undersample_rate))
                loss = loss * cost_mask.float()
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target, threshold=None):
        if threshold is None:
            threshold = self.threshold
        metric = {}
        _target = target["label"]
        _labeled = ~torch.isnan(_target) & target["mask"]
        _size = functional.variadic_sum(_labeled.long(), target["size"]) .view(-1)
        
        pred_binary = (pred[_labeled] > threshold).long().cpu()
        tp, tn, fp, fn = self._calculate_base_metrics(pred_binary, _target[_labeled].long().cpu())
        
        for _metric in self.metric:
            if _metric == "micro_acc":
                score = metrics.accuracy(pred[_labeled], _target[_labeled].long())
            elif _metric == "micro_auroc":
                score = metrics.area_under_roc(pred[_labeled], _target[_labeled])
            elif _metric == "micro_auprc":
                score = metrics.area_under_prc(pred[_labeled], _target[_labeled])
            elif _metric == "macro_auroc":
                score = metrics.variadic_area_under_roc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_auprc":
                score = metrics.variadic_area_under_prc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_acc":
                score = pred[_labeled].argmax(-1) == _target[_labeled]
                score = functional.variadic_mean(score.float(), _size).mean()
            elif _metric == "mcc":
                target_inp = _target[_labeled].long().cpu()
                score = mcc_sklearn(pred_binary, target_inp)
            elif _metric == "sensitivity":
                score = tp / (tp + fn)
            elif _metric == "specificity":
                score = tn / (tn + fp)
            elif _metric == "accuracy":
                score = (tp + tn) / (tp + tn + fp + fn)
            elif _metric == "precision":
                score = tp / (tp + fp)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric) if _metric != 'mcc' else 'mcc'
            metric[name] = score

        return metric

    def _calculate_base_metrics(self, pred, target):
        tp = (pred * target).sum()
        tn = ((1 - pred) * (1 - target)).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        assert(tp + tn + fp + fn == len(pred))
        return tp, tn, fp, fn
    

class MeanEnsembleNodePropertyPrediction(NodePropertyPrediction):
    def __init__(self, model, state_dict_files, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.state_dict_files = state_dict_files
    
    def predict(self, batch, all_loss=None, metric=None):
        mean_prediction = None
        for model_file in self.state_dict_files:
            self.load_state_dict(torch.load(model_file, map_location=self.device), strict=False)
            current_prediction = super(MeanEnsembleNodePropertyPrediction, self).predict(batch, all_loss, metric)
            if mean_prediction is None:
                mean_prediction = current_prediction
            else:
                mean_prediction += current_prediction
        mean_prediction /= len(self.state_dict_files)

        return mean_prediction
