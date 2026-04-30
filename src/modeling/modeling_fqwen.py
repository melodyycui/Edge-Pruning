# coding=utf-8
"""PyTorch Qwen2 model with Edge Pruning."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging, ModelOutput
from transformers import Qwen2Config, AutoTokenizer
from dataclasses import dataclass

from l0_fllama import deterministic_z_from_log_alpha, sample_z_from_log_alpha

logger = logging.get_logger(__name__)

#UTIL FUNCTIONS -----------------------------------------------------------------------

def num_writers(config, with_embedding_nodes=False):
    # The token and position embeddings are writers, if they exist
    n_writers = 1 if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        # Each attention head is a writer, as is the MLP
        n_writers += config.num_attention_heads + 1
    
    return n_writers

def num_readers(config):
    # The number of readers does not depend on whether the model has embedding nodes
    n_readers = 0
    for l in range(config.num_hidden_layers):
        # Each attention head Q/K/V is a reader, as is the MLP
        n_readers += (config.num_attention_heads + 2 * config.num_key_value_heads) + 1
    # There is a final read
    n_readers += 1
    return n_readers

def num_edges(config, with_embedding_nodes=False):
    # If there are embedding nodes, they write to all readers
    n_edges = num_readers(config) if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        # Each attention head writes to this layer's MLP, (MLP + head Q/K/Vs) of future layers and the final read
        n_edges += config.num_attention_heads * (
            1 + 
            (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) + 
            1
        )
        
        # The MLP writes to (MLP + head Q/K/Vs) of future layers and the final read
        n_edges += (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) + 1
    
    return n_edges

def num_nodes(config, with_embedding_nodes=False):
    return num_writers(config, with_embedding_nodes)

def writer_idx_to_name(writer_idx, num_layers, num_heads, with_embedding_nodes=False):
    if with_embedding_nodes:
        if writer_idx == 0:
            return "embeds"
        else:
            writer_idx -= 1
    
    layer_idx = writer_idx // (num_heads + 1)
    head_idx = writer_idx % (num_heads + 1)
    if head_idx == num_heads:
        return f"m{layer_idx}"
    else:
        return f"a{layer_idx}.h{head_idx}"

def writer_name_to_idx(name, num_layers, num_heads, with_embedding_nodes=False):
    idx = 0
    if with_embedding_nodes:
        if name == "embeds":
            return 0
        else:
            idx += 1
    if name.startswith("m"):
        layer_idx = int(name[1:])
        idx += layer_idx * (num_heads + 1) + num_heads
    elif name.startswith("a"):
        parts = name.split(".")
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        idx += layer_idx * (num_heads + 1) + head_idx
    else:
        raise ValueError(f"Unrecognized writer name {name}")
    return idx
    
def reader_idx_to_name(reader_idx, num_layers, num_heads, num_key_value_heads):
    layer_idx = reader_idx // (num_heads + 2 * num_key_value_heads + 1)
    head_idx = reader_idx % (num_heads + 2 * num_key_value_heads + 1)
    if layer_idx == num_layers:
        return "resid_post"
    
    if head_idx < num_heads:
        return f"a{layer_idx}.h{head_idx}.q"
    elif head_idx < num_heads + num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads}.k"
    elif head_idx < num_heads + 2 * num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads - num_key_value_heads}.v"
    else:
        return f"m{layer_idx}"

def get_mask(log_alpha, training=False, threshold_for_deterministic=None, apply_one=False):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask

# RMSNORM and ROTARY EMBEDDING (COPIED W SMALL EDITS FROM FLLAMA CODE) -------------------------

class FQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ALL_LAYERNORM_LAYERS.append(FQwen2RMSNorm)


class FQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
# MLP (MOSTLY COPIED FROM FLLAMA)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class FQwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_parameters.get('rope_theta', 10000.0)
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = FQwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _apply_headwise_linear(self, x, weight, bias, num_heads):
        # x is (num_heads, batch_size, seq_len, hidden_dim)
        _, bsz, seq_len, _ = x.shape
        weight_ = weight.view(num_heads, self.head_dim, self.hidden_size)
        projected = torch.einsum('nbld,nhd->nblh', x, weight_)
        if bias is not None:
            projected = projected + bias.view(num_heads, 1, 1, self.head_dim)
        projected = projected.permute(1, 0, 2, 3)  # (batch_size, n_heads, seq_len, head_dim)
        return projected

    def _apply_output_linear(self, x, weight, num_heads):
        # x is (batch_size, num_heads, seq_len, head_dim)
        weight_ = weight.view(self.hidden_size, num_heads, self.head_dim)
        projected = torch.einsum('bnlh,dnh->nbld', x, weight_)
        return projected

    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        _, bsz, q_len, _ = q_hidden_states.size()

        query_states = self._apply_headwise_linear(
            q_hidden_states, self.q_proj.weight, self.q_proj.bias, self.num_heads
        )
        key_states = self._apply_headwise_linear(
            k_hidden_states, self.k_proj.weight, self.k_proj.bias, self.num_key_value_heads
        )
        value_states = self._apply_headwise_linear(
            v_hidden_states, self.v_proj.weight, self.v_proj.bias, self.num_key_value_heads
        )

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = self._apply_output_linear(attn_output, self.o_proj.weight, self.num_heads)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class FQwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        with_embedding_nodes: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = FQwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = FQwen2MLP(config)
        self.input_layernorm = FQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.mlp.gate_proj.weight.dtype

        writer_offset = 1 if with_embedding_nodes else 0
        self.attn_writer_idx = writer_offset + layer_idx * (self.num_heads + 1)
        self.attn_reader_idx = layer_idx * (self.num_heads + 2 * self.num_kv_heads + 1)
        self.mlp_writer_idx = writer_offset + (layer_idx + 1) * (self.num_heads + 1) - 1
        self.mlp_reader_idx = (layer_idx + 1) * (self.num_heads + 2 * self.num_kv_heads + 1) - 1

        self.q_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.k_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.v_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.attn_write_log_alphas = nn.Parameter(torch.empty(self.num_heads, dtype=self._dtype))
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        attn_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        attn_read_common_mask[:self.attn_writer_idx] = 1
        attn_read_common_mask = attn_read_common_mask.unsqueeze(1)
        self.register_buffer("attn_read_common_mask", attn_read_common_mask)

        attn_write_common_mask = F.pad(
            torch.eye(self.num_heads, dtype=torch.float32).to(self._dtype),
            (self.attn_writer_idx, self.num_writers - self.attn_writer_idx - self.num_heads, 0, 0)
        )
        self.register_buffer("attn_write_common_mask", attn_write_common_mask)

        self.mlp_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.mlp_write_log_alphas = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        mlp_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        mlp_read_common_mask[:self.mlp_writer_idx] = 1
        self.register_buffer("mlp_read_common_mask", mlp_read_common_mask)

        mlp_write_common_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        mlp_write_common_mask[self.mlp_writer_idx, 0] = 1
        self.register_buffer("mlp_write_common_mask", mlp_write_common_mask)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self):
        z_q = get_mask(self.q_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z_q = z_q[:self.attn_writer_idx, :]
        z_k = get_mask(self.k_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z_k = z_k[:self.attn_writer_idx, :]
        z_v = get_mask(self.v_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z_v = z_v[:self.attn_writer_idx, :]
        z_mlp = get_mask(self.mlp_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z_mlp = z_mlp[:self.mlp_writer_idx]
        return (z_q, z_k, z_v, z_mlp)

    @torch.no_grad()
    def get_node_masks(self):
        z_attn = get_mask(self.attn_write_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_mlp = get_mask(self.mlp_write_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic).reshape([])
        return (z_attn, z_mlp)

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

    def attn_read(self, x, corr_x=None, embeds=None):
        q_m = get_mask(self.q_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        k_m = get_mask(self.k_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        v_m = get_mask(self.v_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)

        q_z = q_m * self.attn_read_common_mask
        k_z = k_m * self.attn_read_common_mask
        v_z = v_m * self.attn_read_common_mask

        x_q = torch.einsum("wbsd,wh->hbsd", x, q_z)
        x_k = torch.einsum("wbsd,wh->hbsd", x, k_z)
        x_v = torch.einsum("wbsd,wh->hbsd", x, v_z)

        if embeds is not None:
            x_q = x_q + embeds.unsqueeze(0)
            x_k = x_k + embeds.unsqueeze(0)
            x_v = x_v + embeds.unsqueeze(0)

        if corr_x is not None:
            x_q = x_q + torch.einsum("wbsd,wh->hbsd", corr_x, (1-q_m) * self.attn_read_common_mask)
            x_k = x_k + torch.einsum("wbsd,wh->hbsd", corr_x, (1-k_m) * self.attn_read_common_mask)
            x_v = x_v + torch.einsum("wbsd,wh->hbsd", corr_x, (1-v_m) * self.attn_read_common_mask)

        z_edges_sum = torch.sum(q_z) + torch.sum(k_z) + torch.sum(v_z)
        return x_q, x_k, x_v, z_edges_sum

    def attn_write(self, residual, x, corr_x=None):
        z = get_mask(self.attn_write_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic).reshape(-1, 1, 1, 1)
        x = x * z
        if corr_x is not None:
            x = x + corr_x[self.attn_writer_idx : self.attn_writer_idx + self.num_heads] * (1-z)
        x = torch.einsum("nbsd,nw->wbsd", x, self.attn_write_common_mask)
        residual = residual + x
        z_nodes_sum = torch.sum(z)
        return residual, z_nodes_sum

    def mlp_read(self, x, corr_x=None, embeds=None):
        m = get_mask(self.mlp_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z = m * self.mlp_read_common_mask
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-m) * self.mlp_read_common_mask)
        z_edges_sum = torch.sum(z)
        return x_z, z_edges_sum

    def mlp_write(self, residual, x, corr_x=None):
        z = get_mask(self.mlp_write_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic).reshape(1, 1, 1)
        x = x * z
        if corr_x is not None:
            x = x + corr_x[self.mlp_writer_idx] * (1-z)
        x = torch.einsum("ibsd,wi->wbsd", x.unsqueeze(0), self.mlp_write_common_mask)
        residual = residual + x
        return residual, torch.sum(z)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        corr_x: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        q_hidden_states, k_hidden_states, v_hidden_states, z_attn_edges_sum = self.attn_read(hidden_states, corr_x=corr_x, embeds=embeds)
        q_hidden_states = self.input_layernorm(q_hidden_states)
        k_hidden_states = self.input_layernorm(k_hidden_states)
        v_hidden_states = self.input_layernorm(v_hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            q_hidden_states=q_hidden_states,
            k_hidden_states=k_hidden_states,
            v_hidden_states=v_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        residual, z_attn_nodes_sum = self.attn_write(residual, hidden_states, corr_x=corr_x)

        hidden_states, z_mlp_edges_sum = self.mlp_read(residual, corr_x=corr_x, embeds=embeds)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states, z_mlp_nodes_sum = self.mlp_write(residual, hidden_states, corr_x=corr_x)

        z_edges_sum = z_attn_edges_sum + z_mlp_edges_sum
        z_nodes_sum = z_attn_nodes_sum + z_mlp_nodes_sum

        outputs = (hidden_states, z_edges_sum, z_nodes_sum)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

@dataclass
class FQwen2ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


@dataclass
class FQwen2ForCausalLMOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


class FQwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FQwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class FQwen2Model(FQwen2PreTrainedModel):
    def __init__(
        self,
        config: Qwen2Config,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                FQwen2DecoderLayer(config, layer_idx, with_embedding_nodes=with_embedding_nodes)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.num_layers = config.num_hidden_layers
        self.num_edges = num_edges(config, with_embedding_nodes=with_embedding_nodes)
        self.num_nodes = num_nodes(config, with_embedding_nodes=with_embedding_nodes)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.norm.weight.dtype
        self.with_embedding_nodes = with_embedding_nodes

        if self.with_embedding_nodes:
            self.token_write_log_alpha = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)

            token_write_mask = torch.zeros(self.num_writers, dtype=self._dtype)
            token_write_mask[0] = 1
            self.register_buffer("token_write_mask", token_write_mask)

        self.final_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)

        if disable_linear_regularization_term:
            self.sparsity_lambda_edges_1 = torch.tensor([0.0], dtype=self._dtype)
            self.sparsity_lambda_nodes_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_edges_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.sparsity_lambda_nodes_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_edges_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_nodes_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        for layer in self.layers:
            layer.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic
        for layer in self.layers:
            layer.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_edge_masks())
        z_final = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        masks.append((z_final,))
        return masks

    @torch.no_grad()
    def get_node_masks(self):
        masks = []
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape([])
            masks.append((z_tokens,))
        for layer in self.layers:
            masks.append(layer.get_node_masks())
        return masks

    @torch.no_grad()
    def get_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_
        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_
        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_node_sparsity(self):
        node_masks = self.get_node_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        if self.with_embedding_nodes:
            s_, n_ = process(node_masks[0][0])
            s += s_
            n += n_
            offset = 1
        else:
            offset = 0
        for l in range(len(self.layers)):
            for i in range(2):
                s_, n_ = process(node_masks[l+offset][i])
                s += s_
                n += n_
        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()
        full_node_mask = torch.cat([mask.reshape(-1) for group in node_masks for mask in group], dim=0)
        def process(mask):
            mask = mask * full_node_mask[:mask.shape[0]].reshape(-1, *([1] * (mask.ndim - 1)))
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_
        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_
        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_edges(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()
        allowed_writers = []
        edges = []
        if self.with_embedding_nodes:
            if node_masks[0][0] == 1:
                allowed_writers.append(0)
            offset = 1
            layer_offset = 1
        else:
            offset = 0
            layer_offset = 0
        for l in range(self.num_layers):
            attn_writers = node_masks[l+layer_offset][0]
            for i in range(self.num_heads):
                if attn_writers[i] == 1:
                    allowed_writers.append(offset + l * (1 + self.num_heads) + i)
            mlp_writers = node_masks[l+layer_offset][1]
            if mlp_writers == 1:
                allowed_writers.append(offset + (l+1) * (1 + self.num_heads) - 1)
            attn_q_edges, attn_k_edges, attn_v_edges, mlp_edges = edge_masks[l]
            for from_idx in range(attn_q_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                for head_no in range(attn_q_edges.shape[1]):
                    if attn_q_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_k_edges.shape[1]):
                    if attn_k_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_v_edges.shape[1]):
                    if attn_v_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + self.num_kv_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
            for from_idx in range(mlp_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                if mlp_edges[from_idx] == 1:
                    to_idx = (l+1) * (1 + self.num_heads + 2 * self.num_kv_heads) - 1
                    edges.append((
                        writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes),
                        reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                    ))
        final_read_mask = edge_masks[self.num_layers][0]
        for from_idx in range(self.num_writers):
            if (from_idx in allowed_writers) and (final_read_mask[from_idx] == 1):
                edges.append((
                    writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes),
                    f"resid_post"
                ))
        return edges

    @torch.no_grad()
    def reset_all_log_alphas(self):
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        for layer in self.layers:
            layer.reset_all_log_alphas()
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.sparsity_lambda_edges_1.data.zero_()
        self.sparsity_lambda_nodes_1.data.zero_()

    def read(self, x, corr_x=None, embeds=None):
        z = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-z))
        z_edges_sum = torch.sum(z)
        return x_z, z_edges_sum

    def write(self, tok_embeds, corr_x=None):
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape(1, 1, 1)
            tok_embeds = tok_embeds * z_tokens
            if corr_x is not None:
                tok_embeds = tok_embeds + corr_x[0] * (1 - z_tokens)
            hidden_states = tok_embeds.detach().unsqueeze(0) * self.token_write_mask.reshape(-1, 1, 1, 1)
            z_nodes_sum = torch.sum(z_tokens)
            return hidden_states, None, z_nodes_sum
        else:
            hidden_states = torch.zeros(self.num_writers, *tok_embeds.shape, dtype=tok_embeds.dtype, device=tok_embeds.device)
            z_nodes_sum = 0
            return hidden_states, tok_embeds, z_nodes_sum

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x=None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwen2ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache()
                past_seen_tokens = 0

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens + inputs_embeds.shape[1]
        )

        hidden_states, embeds, z_nodes_sum = self.write(inputs_embeds, corr_x=corr_x)
        z_edges_sum = 0

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    corr_x,
                    embeds,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    corr_x=corr_x,
                    embeds=embeds,
                )

            hidden_states, z_layer_edges_sum, z_layer_nodes_sum = layer_outputs[0], layer_outputs[1], layer_outputs[2]
            z_edges_sum = z_edges_sum + z_layer_edges_sum
            z_nodes_sum = z_nodes_sum + z_layer_nodes_sum

            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        if output_writer_states:
            writer_states = hidden_states
        else:
            writer_states = None

        hidden_states, z_final_edges_sum = self.read(hidden_states, corr_x=corr_x, embeds=embeds)
        z_edges_sum = z_edges_sum + z_final_edges_sum
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        model_edge_sparsity = 1 - (z_edges_sum / self.num_edges)
        model_node_sparsity = 1 - (z_nodes_sum / self.num_nodes)

        if target_edge_sparsity is None:
            edge_loss = None
        else:
            edge_loss = self.sparsity_lambda_edges_1.reshape([]) * (
                model_edge_sparsity - target_edge_sparsity
            ) + self.sparsity_lambda_edges_2.reshape([]) * (
                model_edge_sparsity - target_edge_sparsity
            )**2

        if target_node_sparsity is None:
            node_loss = None
        else:
            node_loss = self.sparsity_lambda_nodes_1.reshape([]) * (
                model_node_sparsity - target_node_sparsity
            ) + self.sparsity_lambda_nodes_2.reshape([]) * (
                model_node_sparsity - target_node_sparsity
            )**2

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if target_edge_sparsity is not None:
            target_edge_sparsity = torch.tensor(target_edge_sparsity, device=model_edge_sparsity.device, dtype=model_edge_sparsity.dtype)
        if target_node_sparsity is not None:
            target_node_sparsity = torch.tensor(target_node_sparsity, device=model_node_sparsity.device, dtype=model_node_sparsity.dtype)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states, next_cache, all_hidden_states, all_self_attns,
                    writer_states, target_edge_sparsity, target_node_sparsity,
                    model_edge_sparsity, model_node_sparsity, edge_loss, node_loss,
                ] if v is not None
            )

        return FQwen2ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            writer_states=writer_states,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            model_edge_sparsity=model_edge_sparsity,
            model_node_sparsity=model_node_sparsity,
            edge_loss=edge_loss,
            node_loss=node_loss,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, current_length):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):
            target_length = self.config.max_position_embeddings
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else current_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class FQwen2ForCausalLM(FQwen2PreTrainedModel):

    def __init__(
        self,
        config: Qwen2Config,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
    ):
        super().__init__(config)
        self.model = FQwen2Model(
            config,
            with_embedding_nodes=with_embedding_nodes,
            disable_linear_regularization_term=disable_linear_regularization_term
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.model.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.model.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        return self.model.get_edge_masks()

    @torch.no_grad()
    def get_node_masks(self):
        return self.model.get_node_masks()

    @torch.no_grad()
    def get_edge_sparsity(self):
        return self.model.get_edge_sparsity()

    @torch.no_grad()
    def get_node_sparsity(self):
        return self.model.get_node_sparsity()

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        return self.model.get_effective_edge_sparsity()

    @torch.no_grad()
    def get_edges(self):
        return self.model.get_edges()

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.model.reset_all_log_alphas()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x=None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwen2ForCausalLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            corr_x=corr_x,
            output_writer_states=output_writer_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FQwen2ForCausalLMOutput(
            lm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            writer_states=outputs.writer_states,
            target_edge_sparsity=outputs.target_edge_sparsity,
            target_node_sparsity=outputs.target_node_sparsity,
            model_edge_sparsity=outputs.model_edge_sparsity,
            model_node_sparsity=outputs.model_node_sparsity,
            edge_loss=outputs.edge_loss,
            node_loss=outputs.node_loss,
        )


if __name__ == '__main__':
    class FQwen2ForCausalLM(FQwen2PreTrainedModel):

        def __init__(
            self,
            config: Qwen2Config,
            with_embedding_nodes: bool = False,
            disable_linear_regularization_term=False,
        ):
            super().__init__(config)
            self.model = FQwen2Model(
                config,
                with_embedding_nodes=with_embedding_nodes,
                disable_linear_regularization_term=disable_linear_regularization_term
            )
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def get_decoder(self):
            return self.model

        @torch.no_grad()
        def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
            self.model.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

        @torch.no_grad()
        def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
            self.model.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

        @torch.no_grad()
        def get_edge_masks(self):
            return self.model.get_edge_masks()

        @torch.no_grad()
        def get_node_masks(self):
            return self.model.get_node_masks()

        @torch.no_grad()
        def get_edge_sparsity(self):
            return self.model.get_edge_sparsity()

        @torch.no_grad()
        def get_node_sparsity(self):
            return self.model.get_node_sparsity()

        @torch.no_grad()
        def get_effective_edge_sparsity(self):
            return self.model.get_effective_edge_sparsity()

        @torch.no_grad()
        def get_edges(self):
            return self.model.get_edges()

        @torch.no_grad()
        def reset_all_log_alphas(self):
            self.model.reset_all_log_alphas()

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            target_edge_sparsity: Optional[float] = None,
            target_node_sparsity: Optional[float] = None,
            corr_x=None,
            output_writer_states: Optional[bool] = False,
        ) -> Union[Tuple, FQwen2ForCausalLMOutput]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                target_edge_sparsity=target_edge_sparsity,
                target_node_sparsity=target_node_sparsity,
                corr_x=corr_x,
                output_writer_states=output_writer_states,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return FQwen2ForCausalLMOutput(
                lm_loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                writer_states=outputs.writer_states,
                target_edge_sparsity=outputs.target_edge_sparsity,
                target_node_sparsity=outputs.target_node_sparsity,
                model_edge_sparsity=outputs.model_edge_sparsity,
                model_node_sparsity=outputs.model_node_sparsity,
                edge_loss=outputs.edge_loss,
                node_loss=outputs.node_loss,
            )


if __name__ == '__main__':
    from transformers import Qwen2Config, AutoTokenizer, Qwen2ForCausalLM
    model = FQwen2ForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B',
        with_embedding_nodes=True
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    inputs = tokenizer("Hi, my name is", return_tensors="pt")
    model.train()
    outputs = model(**inputs)
    print("Success! Logits shape:", outputs.logits.shape)
    model = FQwen2ForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B',
        with_embedding_nodes=True
    )


    original = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
    original.eval()
    model.eval()
    model.reset_all_log_alphas()

    # Set all edge thresholds to use deterministic mode with threshold 0
    # This means all masks = 1 (keep all edges)
    model.set_edge_threshold_for_deterministic(0.0)
    model.set_node_threshold_for_deterministic(0.0)

    with torch.no_grad():
        original_logits = original(**inputs).logits
        fqwen_logits = model(**inputs).logits

    print("Max diff:", (original_logits.float() - fqwen_logits.float()).abs().max().item())
    print("Any NaN in fqwen?", torch.isnan(fqwen_logits).any().item())
    print("Any NaN in original?", torch.isnan(original_logits).any().item())

    # Check if the issue is in the bias handling in _apply_headwise_linear
    # Print the q_proj bias of both models
    print("Original q_proj bias:", original.model.layers[0].self_attn.q_proj.bias[:5])
    print("FQwen q_proj bias:", model.model.layers[0].self_attn.q_proj.bias[:5])

    # Check the state dict keys for attention
orig_keys = [k for k in original.state_dict().keys() if 'layers.0.self_attn' in k]
fqwen_keys = [k for k in model.state_dict().keys() if 'layers.0.self_attn' in k]
print("Original keys:", orig_keys)
print("FQwen keys:", fqwen_keys)