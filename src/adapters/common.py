from typing import NamedTuple, Union, Callable

import torch.nn as nn


class AdapterConfig(NamedTuple):
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float
    label_emb_size: int
    latent_size: int
    class_num: int
    n_layer: int
    init: str


def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model

def init_lisa_params(module):
    std = 1e-20
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def init_bias_mlp(module):
    std = 1e-2
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()

def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def init_zero_weights(module):
    if isinstance(module, nn.Embedding):
        nn.init.constant_(module.weight, 0.0)