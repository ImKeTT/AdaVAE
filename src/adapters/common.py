from typing import NamedTuple, Union, Callable

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from transformers.modeling_utils import Conv1D


class AdapterConfig(NamedTuple):
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float
    latent_size: int
    class_num: int
    encoder_n_layer: int
    decoder_n_layer: int
    dis_emb: int
    init: str
    adapter_scalar: str
    ffn_option: str # sequential / parallel_attn / parallel_ffn / pfeiffer
    attn_mode: str # 'prefix', 'adapter', 'lora', 'none'
    latent_gen: str # 'averaged_attn', 'linear'
    attn_option: str
    mid_dim: int # prefix middle dimension
    attn_bn: int
    prefix_dropout: float
    tune_enc: bool
    tune_dec: bool
    add_z2adapters: bool ## wether infuse z to adapter blocks


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

######## model layer utils ########
# copied from LoRA: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_init: str="lora",
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.lora_init = lora_init
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.ef_lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.ef_lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'ef_lora_A'):
            if self.lora_init == "bert":
                nn.init.normal_(self.ef_lora_A, std=0.02)
                nn.init.normal_(self.ef_lora_B, std=0.02)
            elif self.lora_init == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.ef_lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.ef_lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.ef_lora_B @ self.ef_lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.ef_lora_B @ self.ef_lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.ef_lora_A.T @ self.ef_lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LoRA_Conv1D(Conv1D, LoRALayer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf: int, nx: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 fan_in_fan_out: bool = False,
                 # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 merge_weights: bool = True,
                 lora_init: str = "lora"
                 ):
        Conv1D.__init__(self, nf, nx)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

        self.lora_init = lora_init
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.ef_lora_A = nn.Parameter(self.weight.new_zeros((r, nx)))
            self.ef_lora_B = nn.Parameter(self.weight.new_zeros((nf, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Adapter_Layer(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        # self.non_linearity = args.non_linearity  # use ReLU by default

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

## Prefix
class Prefix(nn.Module):
    def __init__(self, ada_config, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.n_embd = config.n_embd
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = ada_config.mid_dim
        self.attn_bn = ada_config.attn_bn
        self.prefix_dropout = ada_config.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        # self.control_trans_enc = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
        #
        # self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        # self.control_trans2 = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="cuda"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # temp_control2 = self.wte2(input_tokens)
        # past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        # past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
        #                                          self.match_n_embd)
        # past_key_values2 = self.dropout(past_key_values2)
        # past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)
        #
        # input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        # temp_control_enc = self.wte_enc(input_tokens_enc)
        # past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        # bsz_enc, seqlen, _ = past_key_values_enc.shape
        # past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
        #                                                self.match_n_embd)
        # past_key_values_enc = self.dropout(past_key_values_enc)
        # past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  }
            result.append(temp_dict)
        return result


class PrefixCrossAttn(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'encoder_decoder': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }
            key_val2 = past_key_values2[i]
            temp_dict['self'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                        }
            result.append(temp_dict)
        return result


class PrefixDirectInit(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)
        self.input_tokens = torch.arange(self.attn_bn).long()
        self.encoder_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.encoder_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                               for _ in range(self.match_n_layer)])
        self.decoder_self_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])
        self.decoder_self_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                    for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        self.decoder_cross_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        self.apply(init_bert_weights)

    def _shape(self, x, bsz):
        y = x.view(bsz, self.attn_bn, self.match_n_head, self.match_n_embd)
        y = y.permute([0, 2, 1, 3])
        y = y.contiguous().view(bsz * self.match_n_head, -1, self.match_n_embd)
        return y

    def forward(self, bsz, nsamples=1, device="cuda"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)

        result = []
        for i, (enc_attn_k, enc_attn_v, dec_self_attn_k, dec_self_attn_v, dec_xattn_k, dec_xattn_v) in \
                enumerate(zip(self.encoder_attn_key, self.encoder_attn_value, self.decoder_self_attn_key,
                              self.decoder_self_attn_value, self.decoder_cross_attn_key, self.decoder_cross_attn_value)):
            temp_dict = {'self': {"prev_key": self._shape(dec_self_attn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_self_attn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                         'encoder_decoder': {"prev_key": self._shape(dec_xattn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_xattn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device)  #bsz, attn_bn
                                  },
                         'encoder': {"prev_key": self._shape(enc_attn_k(input_tokens_enc), old_bsz),
                                  "prev_value": self._shape(enc_attn_v(input_tokens_enc), old_bsz),
                                  "prev_key_padding_mask": torch.zeros(old_bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                        }
            result.append(temp_dict)
        return result

############### Network Architechtures ###############
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h


############## Distributions #################
min_epsilon = 1e-5
max_epsilon = 1.-1e-5
#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )

def logisticCDF(x, u, s):
    return 1. / ( 1. + torch.exp( -(x-u) / s ) )

def sigmoid(x):
    return 1. / ( 1. + torch.exp( -x ) )

def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256