from collections import OrderedDict
from torch import Tensor
from transformers import BertModel, BertTokenizer, AutoTokenizer, EsmModel
import torch
from torchdrug import core, models

def separate_alphabets(text):
    separated_text = ""
    for char in text:
        if char.isalpha():
            separated_text += char + " "
    return separated_text.strip()


lm_type_map = {
    'bert': (BertModel, BertTokenizer, "Rostlab/prot_bert"),
    'esm-t6': (EsmModel, AutoTokenizer, "facebook/esm2_t6_8M_UR50D"),
    'esm-t12': (EsmModel, AutoTokenizer, "facebook/esm2_t12_35M_UR50D"),
    'esm-t30': (EsmModel, AutoTokenizer, "facebook/esm2_t30_150M_UR50D"),
    'esm-t33': (EsmModel, AutoTokenizer, "facebook/esm2_t33_650M_UR50D"),
    'esm-t36': (EsmModel, AutoTokenizer, "facebook/esm2_t36_3B_UR50D"),
}

class LMGearNetModel(torch.nn.Module, core.Configurable):
    def __init__(self, 
                 gpu,
                 lm_type='bert',
                 gearnet_hidden_dim_size=512,
                 gearnet_hidden_dim_count=6,
                 gearnet_short_cut=True,
                 gearnet_concat_hidden=True,
                 lm_concat_to_output=False,
                 lm_short_cut=False,
    ):
        super().__init__()
        Model, Tokenizer, pretrained_model_name = lm_type_map[lm_type]
        self.tokenizer = Tokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
        self.lm = Model.from_pretrained(pretrained_model_name).to(f'cuda:{gpu}')
        self.gearnet = models.GearNet(
            input_dim=self.lm.config.hidden_size,
            hidden_dims=[gearnet_hidden_dim_size] * gearnet_hidden_dim_count,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=gearnet_concat_hidden,
            short_cut=gearnet_short_cut,
            readout="sum"
        ).to(f'cuda:{gpu}')
        self.input_dim = 21
        self.output_dim = self.gearnet.output_dim + self.lm.config.hidden_size if lm_concat_to_output else self.gearnet.output_dim
        self.lm_concat_to_output = lm_concat_to_output
        if lm_short_cut:
            assert self.gearnet.output_dim == self.lm.config.hidden_size, "lm_short_cut is only available when gearnet output dim is equal to lm hidden size"
        self.lm_short_cut = lm_short_cut
        self.gpu = gpu

    def forward(self, graph, _, all_loss=None, metric=None):
        # print("sequence: ", graph.to_sequence())
        input = [separate_alphabets(seq) for seq in graph.to_sequence()]
        input_len = [len(seq.replace(' ', '')) for seq in input]

        # At large batch size, tokenization becomes the bottleneck
        encoded_input = self.tokenizer(input, return_tensors='pt', padding=True).to(f'cuda:{self.gpu}')
        embedding_rpr = self.lm(**encoded_input)
        
        lm_residue_feature = []
        for i, emb in enumerate(embedding_rpr.last_hidden_state):
            # skip residue feature for [CLS] and [SEP], since they are not in the original sequence
            lm_residue_feature.append(emb[1:1+input_len[i]])
        
        lm_output = torch.cat(lm_residue_feature)

        gearnet_output = self.gearnet(graph, lm_output)

        final_output = torch.cat([gearnet_output['node_feature'], lm_output], dim=-1) if self.lm_concat_to_output else gearnet_output['node_feature']

        if self.lm_short_cut:
            final_output = final_output + lm_output
        return {
            "node_feature": final_output,
        }
    
    def freeze_lm(self, freeze_all=True, freeze_layer_count=None):
        if freeze_all:
            # freeze the entire bert model
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            # freeze the embeddings
            for param in self.lm.embeddings.parameters():
                param.requires_grad = False
            if freeze_layer_count != -1:
                # freeze layers in bert_model.encoder
                for layer in self.lm.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
        
    
    def freeze_gearnet(self, freeze_all=False, freeze_layer_count=0):
        if freeze_all:
            for param in self.gearnet.parameters():
                param.requires_grad = False
        elif freeze_layer_count != 0:
            for layer in self.gearnet.layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.gearnet.edge_layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.gearnet.batch_norms[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False


class GearNetWrapModel(torch.nn.Module, core.Configurable):
    def freeze(self, freeze_all=False, freeze_layer_count=0):
        if freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_layer_count != 0:
            for layer in self.model.layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.model.edge_layers[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.model.batch_norms[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    def __init__(self, input_dim, hidden_dims, gpu, graph_sequential_max_distance=3):
        super().__init__()
        self.gpu = gpu
        self.model = models.GearNet(
            num_relation=7 + (graph_sequential_max_distance - 2) * 2,
            edge_input_dim=59 + (graph_sequential_max_distance - 2) * 2,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).cuda(gpu)
        self.output_dim = self.model.output_dim
        self.input_dim = input_dim

    def forward(self, graph, input, all_loss=None, metric=None):
        return self.model(graph, input, all_loss, metric)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)
