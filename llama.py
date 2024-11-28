import inspect
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_rope_freq():
    return {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }


@dataclass
class Llama3Config:
    vocab_size: int = 32000
    context_length: int = 8192
    emb_dim: int = 2048
    n_heads: int = 32
    n_layers: int = 16
    hidden_dim: int = 8192
    n_kv_groups: int = 8
    rope_base: float = 500_000.0
    dtype: torch.dtype = torch.bfloat16
    rope_freq: dict = field(default_factory=default_rope_freq)


class FeedForward(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.fc1 = nn.Linear(
            config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False
        )
        self.fc2 = nn.Linear(
            config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False
        )
        self.fc3 = nn.Linear(
            config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


def precompute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, freq_config=None
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(
        context_length, head_dim, rope_base, freq_config, dtype=torch.float32
    ):
        key = (
            context_length,
            head_dim,
            rope_base,
            tuple(freq_config.values()) if freq_config else freq_config,
            dtype,
        )

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(
                head_dim, rope_base, context_length, freq_config
            )
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        num_kv_groups,
        rope_base=10_000,
        rope_config=None,
        dtype=None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype
        )
        self.register_buffer("mask", mask)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(
            1, 2
        )  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(
            self.group_size, dim=1
        )  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(
            self.group_size, dim=1
        )  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            num_kv_groups=config.n_kv_groups,
            rope_base=config.rope_base,
            rope_config=config.rope_freq,
            dtype=config.dtype,
        )
        self.ff = FeedForward(config)
        self.norm1 = nn.RMSNorm(config.emb_dim, eps=1e-5)
        self.norm2 = nn.RMSNorm(config.emb_dim, eps=1e-5)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut  # Add the original input back

        return x


class Llama3(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.emb_dim, dtype=config.dtype
        )

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.RMSNorm(config.emb_dim, eps=1e-5)
        self.out_head = nn.Linear(
            config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype
        )

    def forward(self, in_idx, targets=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self, weight_decay, learning_rate, device_type, master_process=True
    ):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
