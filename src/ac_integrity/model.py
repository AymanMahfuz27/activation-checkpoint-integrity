"""Decoder-only next-token model with explicit audit attention and tied weights."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.epsilon) * self.weight


def rotate(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.width // config.heads
        kv_width = config.kv_heads * self.head_dim
        self.q = nn.Linear(config.width, config.width, bias=False)
        self.k = nn.Linear(config.width, kv_width, bias=False)
        self.v = nn.Linear(config.width, kv_width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        c = self.config
        q = self.q(x).view(batch, length, c.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, c.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, c.kv_heads, self.head_dim).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(c.heads // c.kv_heads, dim=1)
        v = v.repeat_interleave(c.heads // c.kv_heads, dim=1)
        if c.attention == "sdpa":
            context = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                     dropout_p=c.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            probabilities = F.softmax(scores, dim=-1)
            probabilities = F.dropout(probabilities, c.dropout, self.training)
            context = probabilities @ v
        return self.output(context.transpose(1, 2).contiguous().view(batch, length, c.width))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.width, config.intermediate, bias=False)
        self.up = nn.Linear(config.width, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width, config.norm_eps)
        self.feed_forward = FeedForward(config)
        self.natural_randomness = False

    def forward(self, x, cos, sin):
        h = x + self.attention(self.attention_norm(x), cos, sin)
        update = self.feed_forward(self.ffn_norm(h))
        # The legitimate mode in #84864 replaces aten.rand during forward.
        # This operation is identical in trigger-on and trigger-off arms.
        if self.natural_randomness:
            update = update * torch.rand(update.shape, device=update.device, dtype=update.dtype)
        return h + update


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config.model
        self.embedding = nn.Embedding(m.vocab_size, m.width)
        self.blocks = nn.ModuleList([Block(m) for _ in range(m.layers)])
        self.norm = RMSNorm(m.width, m.norm_eps)
        self.output = nn.Linear(m.width, m.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        self.runtime = None
        for block in self.blocks:
            block.natural_randomness = config.adapter.name == "pytorch_84864"

    def forward(self, tokens):
        m = self.config.model
        head_dim = m.width // m.heads
        frequencies = 1.0 / (m.rope_theta ** (torch.arange(0, head_dim, 2, device=tokens.device).float() / head_dim))
        angles = torch.arange(tokens.shape[1], device=tokens.device).float().unsqueeze(1) * frequencies
        cos, sin = angles.cos(), angles.sin()
        x = self.embedding(tokens)
        c = self.config.checkpoint
        for index, block in enumerate(self.blocks):
            if c.enabled and (not c.blocks or index in c.blocks):
                options = {}
                if self.runtime is not None:
                    options["context_fn"] = self.runtime.checkpoint_contexts(f"blocks.{index}")
                with set_checkpoint_early_stop(False):
                    x = checkpoint(block, x, cos, sin, use_reentrant=False,
                                   preserve_rng_state=c.preserve_rng_state,
                                   determinism_check=c.determinism_check, **options)
            else:
                x = block(x, cos, sin)
        return self.output(self.norm(x))
