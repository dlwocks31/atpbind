from transformers import BertModel, BertTokenizer
import torch
from torchdrug import core, models


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


def separate_alphabets(text):
    separated_text = ""
    for char in text:
        if char.isalpha():
            separated_text += char + " "
    return separated_text.strip()

class LMGearNetModel(torch.nn.Module, core.Configurable):
    def __init__(self, gpu, gearnet_hidden_dim_size=512, gearnet_hidden_dim_count=6):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False)
        self.bert_model = BertModel.from_pretrained("Rostlab/prot_bert").to(f'cuda:{gpu}')
        _freeze_bert(self.bert_model, freeze_bert=True)
        self.gearnet = models.GearNet(
            input_dim=1024, #self.bert_model.config.hidden_size,
            hidden_dims=[gearnet_hidden_dim_size] * gearnet_hidden_dim_count,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum"
        ).to(f'cuda:{gpu}')
        self.input_dim = 21
        self.output_dim = self.gearnet.output_dim
        self.gpu = gpu

    def forward(self, graph, _, all_loss=None, metric=None):
        # print("at forward, graph: ", graph)
        # print("sequence: ", graph.to_sequence())
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]

        encoded_input = self.bert_tokenizer(
            input, return_tensors='pt', padding=True).to(f'cuda:{self.gpu}')
        # print("Input size: ", encoded_input["input_ids"].size())
        x = self.bert_model(**encoded_input)
        # print("Output size just after bert model: ", x.last_hidden_state.size())
        
        # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
        lm_output = x.last_hidden_state.squeeze()[1:-1]
        
        # print(f'lm_output shape: {lm_output.shape}')
        gearnet_output = self.gearnet(graph, lm_output)
        return gearnet_output
    