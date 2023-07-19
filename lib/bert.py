# freeze_bert in https://github.com/aws-samples/lm-gvp/blob/0b7a6d96486e2ee222929917570432296554cfe7/lmgvp/modules.py#L47

from transformers import BertModel, BertTokenizer
from torchdrug import core
import torch

def _freeze_bert(
    bert_model: BertModel, freeze_bert=True, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)

    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether or not to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None


# Cusom model Wrapping BERT: check https://torchdrug.ai/docs/notes/model.html
class BertWrapModel(torch.nn.Module, core.Configurable):
    def __init__(self, freeze_bert, freeze_layer_count):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False)
        self.bert_model = BertModel.from_pretrained(
            "Rostlab/prot_bert").to('cuda')
        _freeze_bert(self.bert_model, 
                     freeze_bert=freeze_bert,
                     freeze_layer_count=freeze_layer_count)
        self.input_dim = 21
        self.output_dim = self.bert_model.config.hidden_size

    def forward(self, graph, _, all_loss=None, metric=None):
        input = [seq.replace('.', ' ') for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]

        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.bert_tokenizer(
            input, return_tensors='pt', padding=True).to('cuda')
        embedding_rpr = self.bert_model(**encoded_input)
        
        residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            residue_feature.append(emb[1:1+input_len[i]])
        
        x = torch.cat(residue_feature)

        return {"residue_feature": x}
