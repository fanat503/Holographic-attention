import csv
import json
import math
import random
import time
import gc
import shutil
from contextlib import nullcontext
from dataclasses import dataclass

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps).to(dtype=x.dtype)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # tensor-core friendly
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert (config.n_embd // config.n_head) % 2 == 0, "head_dim must be even"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.phase_mult = config.phase_mult

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.W_phase_q = nn.Parameter(
            torch.empty(self.n_head, config.n_embd, self.head_dim // 2)
        )
        self.W_phase_k = nn.Parameter(
            torch.empty(self.n_head, config.n_embd, self.head_dim // 2)
        )

        nn.init.constant_(self.W_phase_q, 0.0)
        nn.init.constant_(self.W_phase_k, 0.0)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )
        self.last_entropy = None

    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_dim

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        angles_q = torch.einsum("btd,hdk->bthk", x, self.W_phase_q) * self.phase_mult
        angles_k = torch.einsum("btd,hdk->bthk", x, self.W_phase_k) * self.phase_mult

        angles_q = angles_q.permute(0, 2, 1, 3)  # (B, H, T, hs//2)
        angles_k = angles_k.permute(0, 2, 1, 3)

        cos_q, sin_q = torch.cos(angles_q), torch.sin(angles_q)
        cos_k, sin_k = torch.cos(angles_k), torch.sin(angles_k)

        q_real, q_imag = q.chunk(2, dim=-1)
        k_real, k_imag = k.chunk(2, dim=-1)

        q = torch.cat([
            q_real * cos_q - q_imag * sin_q,
            q_real * sin_q + q_imag * cos_q
        ], dim=-1)

        k = torch.cat([
            k_real * cos_k - k_imag * sin_k,
            k_real * sin_k + k_imag * cos_k
        ], dim=-1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(~mask, -1e4)
        att = att - att.max(dim=-1, keepdim=True)[0]
        att = F.softmax(att, dim=-1)

        with torch.no_grad():
            p = att[:min(att.size(0), 2)].clamp(min=1e-9)
            entropy = -(p * p.log()).sum(dim=-1).mean(dim=(0, 2))
            self.last_entropy = entropy.detach().cpu()

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640
    phase_mult: float = 0.15  # baseline: 0.0, holographic: 0.05, holographic-strong: 0.15

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss
