"""学生实现：SwiGLU，并可选用 RoPE 与 RMSNorm。

已完成的等量对照（1200 步、宽 128）：只换 SwiGLU，测试 BPB 2.041。
提交版（configs/submit.json）：再换相对位置编码和 RMSNorm，并加宽加深。
带 dropout 的加长训练见 configs/submit_dropout.json。
默认 pos=learned、norm=layer、dropout=0，以便加载已有 checkpoint。
"""
import torch
from torch import nn
from torch.nn import functional as F


def swiglu_hidden(width):
    """按原 GELU MLP 的权重规模（约 8 * width^2）反推 SwiGLU 中间宽度。"""
    return max(1, int(round(8 * width / 3)))


def make_norm(width, kind):
    if kind == 'layer':
        return nn.LayerNorm(width)
    if kind == 'rms':
        return RMSNorm(width)
    raise ValueError(f'未知的 norm 类型: {kind}（支持 layer / rms）')


class RMSNorm(nn.Module):
    """均方根归一化，只学习缩放，不含偏置。"""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """门控前馈：down(SiLU(gate(x)) * up(x))。"""

    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


def rope_cos_sin(length, head_dim, device, dtype):
    """构造 RoPE 的 cos/sin，形状 [length, head_dim/2]。"""
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(length, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def apply_rope(x, cos, sin):
    """对 [batch, heads, time, dim] 的偶数/奇数维做旋转。"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width=128, heads=4, ffn='swiglu', norm='layer', use_rope=False, dropout=0., qk_norm=False):
        super().__init__()
        if width % heads:
            raise ValueError('width 必须能被 heads 整除。')
        if (width // heads) % 2:
            raise ValueError('每个头的维度必须为偶数，RoPE 才能配对旋转。')
        self.heads = heads
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.attn_dropout = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.norm1, self.norm2 = make_norm(width, norm), make_norm(width, norm)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        head_dim = width // heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None
        if ffn == 'gelu':
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
            )
        elif ffn == 'swiglu':
            self.mlp = SwiGLU(width, swiglu_hidden(width))
        else:
            raise ValueError(f'未知的 ffn 类型: {ffn}（支持 swiglu / gelu）')

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            cos, sin = rope_cos_sin(length, width // self.heads, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # 评测时 dropout 必须为 0，否则同一窗口两次前向会对不上合同测试
        drop_p = self.attn_dropout if self.training else 0.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        x = x + self.resid_drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.resid_drop(self.mlp(self.norm2(x)))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        ffn = config.get('ffn', 'swiglu')
        norm = config.get('norm', 'layer')
        pos = config.get('pos', 'learned')
        if pos not in ('learned', 'rope'):
            raise ValueError(f'未知的 pos 类型: {pos}（支持 learned / rope）')
        self.use_rope = pos == 'rope'
        dropout = float(config.get('dropout', 0.))
        qk_norm = bool(config.get('qk_norm', False))
        self.embed_drop = nn.Dropout(dropout)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = None if self.use_rope else nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], ffn, norm, self.use_rope, dropout, qk_norm)
            for _ in range(config['depth'])
        ])
        self.norm = make_norm(width, norm)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        scale = (2 * config['depth']) ** -0.5
        for block in self.blocks:
            block.proj.weight.data.mul_(scale)
            down = block.mlp.down if hasattr(block.mlp, 'down') else block.mlp[-1]
            down.weight.data.mul_(scale)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """训练接口：未归一化的下一 token logits，形状 [batch, time, vocab]。"""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """评测接口：归一化 log 概率；每次调用独立，不读写跨窗口状态。"""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
