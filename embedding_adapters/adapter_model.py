
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x):
        return x * self.gamma

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

def trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)

def zero_init_linear(m: nn.Linear):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        a, g = self.proj(x).chunk(2, dim=-1)
        return self.dropout(a * F.gelu(g))

class ResidualFFNBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1, ls_init: float = 1e-3, drop_path: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = nn.Sequential(
            GEGLU(dim, hidden, dropout=dropout),
            nn.Linear(hidden, dim)
        )
        zero_init_linear(self.ffn[-1])
        self.ls = LayerScale(dim, init_value=ls_init)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        h = self.ffn(self.ln(x))
        return x + self.dp(self.ls(h))

class ResidualMLPAdapterDeep(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 1536,
        depth: int = 8,
        dropout: float = 0.1,
        ls_init: float = 5e-4,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim, eps=1e-5)
        self.in_fc = nn.Linear(in_dim, width)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist() if depth > 1 else [0.0]
        self.blocks = nn.ModuleList([
            ResidualFFNBlock(width, hidden=width * 4, dropout=dropout, ls_init=ls_init, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.mid_ln = nn.LayerNorm(width, eps=1e-5)
        self.out_fc = nn.Linear(width, out_dim)
        self.out_ln = nn.LayerNorm(out_dim, eps=1e-5)

        trunc_normal_(self.in_fc.weight, std=0.02)
        nn.init.zeros_(self.in_fc.bias)
        trunc_normal_(self.out_fc.weight, std=0.02)
        nn.init.zeros_(self.out_fc.bias)

    def forward(self, x):
        x = self.in_fc(self.in_ln(x))
        for b in self.blocks:
            x = b(x)
        y = self.out_fc(self.mid_ln(x))
        y = self.out_ln(y)
        return F.normalize(y, dim=-1)
