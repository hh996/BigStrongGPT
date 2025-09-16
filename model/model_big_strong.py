from transformers import PretrainedConfig
import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class BigStrongConfig(PretrainedConfig):
    model_type = "BigStrong"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        flash_attn: bool = True,
    ):
        """
        :param dropout: 丢弃率
        :param bos_token_id: 起始token id
        :param eos_token_id: 结束token id
        :param hidden_act: 激活函数
        :param hidden_size: 隐藏层大小
        :param intermediate_size: 中间层大小
        :param max_position_embeddings: 最大位置编码
        :param num_attention_heads: 注意力头数
        :param num_hidden_layers: 隐藏层数
        :param num_key_value_heads: 键值对头数
        :param vocab_size: 词汇表大小
        :param rms_norm_eps: RMSNorm的epsilon
        :param rope_theta: RoPE的theta
        :param flash_attn: 是否使用flash attention
        """
        super().__init__()
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        """
        :param normalized_shape: 输入张量的最后一个维度的大小。例如，如果输入张量的形状为 (N, H)，那么 normalized_shape=H 表示对最后一个维度（即 H 维度）做归一化。
        :param eps: 常数，防止除以零
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        x_normalized = x / torch.sqrt(rms) * self.weight
        return x_normalized

# 计算RoPE旋转位置编码
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    :param dim: 向量维度
    :param end: 序列长度
    :param theta: 旋转位置编码的参数
    """
    # torch.arange(0, dim, 2) 生成 0 到 dim-1 的偶数序列（如 dim=6 时为 [0,2,4]）
    # [: (dim//2)]：确保取前 dim/2 个值（避免维度溢出）
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成从 0 到 end-1 的位置索引
    t = torch.arange(end, device=freqs.device)
    # 位置矩阵 * 频率矩阵
    freqs = torch.outer(t, freqs).float()
    # 拼接cos和sin矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    :param q: 查询向量  (batch_size, num_heads, seq_len, head_dim)
    :param k: 键向量  (batch_size, num_heads, seq_len, head_dim)
    :param cos: cos矩阵  (seq_len, head_dim)
    :param sin: sin矩阵  (seq_len, head_dim)
    """

    def rotate_half(x):
        """
        将向量的后半部分维度取负后再和前半部分维度拼接
        [[[[1, 2, 3, 4]]]] -> [[[[-3, -4, 1, 2]]]]
        """
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    # 应用RoPE旋转位置编码
    q_embed = (q * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(2))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(2))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    :param x: 输入张量  (batch_size, seq_len, num_key_value_heads, head_dim)
    :param n_rep: 重复次数
    """
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    """
    相当于把x的最后一个维度重复n_rep-1次, 例如 n_rep=3, 则
    [[[[1, 2, 3, 4]]]] -> [[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]]
    """
    # 
    return (
        x[:, :, :, None, :] # 增加一个新维度
        .expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim)  # 扩展新维度
        .reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)  # 重塑形状
    )


class Attention(nn.Module):
    def __init__(self, args: BigStrongConfig):
        super().__init__()
        # 注意力头数，表示 Key 和值 Value 的头数量
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV 头需要被重复多少次来匹配 Q 头的数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        :param x: 输入张量  (batch_size, seq_len, hidden_size)
        :param position_embeddings: 位置编码  (seq_len, head_dim)
        :param past_key_value: 过去的键值对  (batch_size, seq_len, num_key_value_heads, head_dim)
        :param use_cache: 是否使用缓存
        :param attention_mask: 注意力掩码  (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        xq, xk = self.q_proj(x), self.k_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(batch_size, 1, 1, -1).expand(
                    batch_size, self.n_local_heads, seq_len, -1
                )
                attn_mask = attn_mask.bool() if attention_mask is not None else None
            # 输出 [batch_size, num_heads, seq_len, head_dim]
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(
                0
            )  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: BigStrongConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 门控投影
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        # 下投影
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        # 上投影
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        # 根据配置中的激活函数选择激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class BigStrongBlock(nn.Module):
    def __init__(self, layer_id: int, config: BigStrongConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        """
        :param hidden_states: 输入张量  (batch_size, seq_len, hidden_size)
        :param position_embeddings: 位置编码  (seq_len, head_dim)
        :param past_key_value: 过去的键值对  (batch_size, seq_len, num_key_value_heads, head_dim)
        :param use_cache: 是否使用缓存
        :param attention_mask: 注意力掩码  (batch_size, seq_len)
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class BigStrongModel(nn.Module):
    def __init__(self, config: BigStrongConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [BigStrongBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        :param input_ids: 输入张量  (batch_size, seq_len)
        :param attention_mask: 注意力掩码  (batch_size, seq_len)
        :param past_key_value: 过去的键值对  (batch_size, seq_len, num_key_value_heads, head_dim)
        :param use_cache: 是否使用缓存
        :param kwargs: 其他参数
        """
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class BigStrongForCausalLLM(PreTrainedModel, GenerationMixin):
    config_class = BigStrongConfig

    def __init__(self, config: BigStrongConfig = None):
        self.config = config or BigStrongConfig()
        super().__init__(self.config)
        self.model = BigStrongModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        """
        :param input_ids: 输入张量  (batch_size, seq_len)
        :param attention_mask: 注意力掩码  (batch_size, seq_len)
        :param past_key_value: 过去的键值对  (batch_size, seq_len, num_key_value_heads, head_dim)
        :param use_cache: 是否使用缓存
        :param logits_to_keep: 控制最终输出的logits（预测概率）需要保留的长度（常用于生成任务）
        :param args: 其他参数
        """
        h, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        # 切片 若 logits_to_keep = 3，则等效于 slice(-3, None)，对于一个列表 [a, b, c, d, e]，使用该切片会得到 [c, d, e]。
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__("last_hidden_state", h)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT
