"""Small FP32 eager language model with explicit checkpoint and fault sites.

Import only after the parent has admitted the actual allocated job. Byte-level
tokens make the real-text corpus independent of tokenizer versions. All six
family-selector assumptions stay explicit: attention is ordinary eager softmax,
GELU is exact, and each block has its own nonreentrant checkpoint region.
"""

import math

import selector as core
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class Block(nn.Module):
    def __init__(self, config, index, fault_controller):
        super().__init__()
        width = config["width"]
        self.index = index
        self.heads = config["heads"]
        self.head_width = width // self.heads
        self.fault_controller = fault_controller
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.expand = nn.Linear(width, config["ff_multiplier"] * width)
        self.contract = nn.Linear(config["ff_multiplier"] * width, width)
        self.register_buffer("causal", torch.ones(config["sequence_length"],
                                                  config["sequence_length"], dtype=torch.bool).tril())

    def forward(self, hidden):
        batch, length, width = hidden.shape
        normalized = self.norm1(hidden)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.causal[:length, :length], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.attention_output(attended)
        expanded = self.expand(self.norm2(hidden))
        # This deterministic intervention is before both the GELU observer and
        # native GELU. The native backward audit independently verifies which
        # values autograd actually consumes; output equality is never inferred.
        self.fault_controller.replay_site(expanded, self.index)
        return hidden + self.contract(F.gelu(expanded, approximate="none"))


class LanguageModel(nn.Module):
    def __init__(self, config, fault_controller):
        super().__init__()
        self.config = dict(config)
        self.token = nn.Embedding(config["vocab_size"], config["width"])
        self.position = nn.Embedding(config["sequence_length"], config["width"])
        self.blocks = nn.ModuleList(Block(config, index, fault_controller)
                                    for index in range(config["layers"]))
        self.output = nn.Linear(config["width"], config["vocab_size"], bias=False)
        self.output.weight = self.token.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        for index, block in enumerate(self.blocks):
            if self.training and torch.is_grad_enabled():
                def contexts(region=str(index)):
                    return core.region_context(region, "original"), core.region_context(region, "recompute")

                # Full replay makes compared fields and native event counts
                # independent of PyTorch's early-stop optimization.
                with set_checkpoint_early_stop(False):
                    hidden = checkpoint(block, hidden, use_reentrant=False,
                                        preserve_rng_state=True, context_fn=contexts)
            else:
                hidden = block(hidden)
        return self.output(hidden)
