## core codes for adapter applying
import logging

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput

from adapters.common import AdapterConfig


logging.basicConfig(level=logging.INFO)


## adapter block
class BertAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(BertAdapter, self).__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        ## initialize down_project weight and bias
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        ## initialize up_project weight and bias
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        ## essentially a ''down mapping'' and an ''up mapping'' process
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected


## add every adaptor block after each layer of a Transformer (before the Layer nomalization)
class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = BertAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        ## instantiate adapter obj
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapt_bert_self_output(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)


def add_bert_adapters(bert_model: BertModel, config: AdapterConfig) -> BertModel:
    ## through multiple adapter modules, use attention output as adaptor input.
    ## Every adapter block requires down and up mappings.
    for layer in bert_model.encoder.layer:
        layer.attention.output = adapt_bert_self_output(config)(layer.attention.output)
        layer.output = adapt_bert_self_output(config)(layer.output)
    return bert_model


def unfreeze_bert_adapters(bert_model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in bert_model.named_modules():
        if isinstance(sub_module, (BertAdapter, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model
