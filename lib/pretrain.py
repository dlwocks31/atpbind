from torchdrug import tasks
from torch.nn import functional as F
class CustomAttributeMasking(tasks.AttributeMasking):
    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy
        
        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        return metric

# class CustomUnsuperVised(tasks.Unsupervised):
#     def evavlua