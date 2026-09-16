import math

import torch
import torch.nn as nn


class EdgeAwareGraphTransformerLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=8, edge_dim=32, ffn_dim=256, dropout=0.05):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_norm2 = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.edge_value = nn.Linear(edge_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, 2 * edge_dim),
            nn.GELU(),
            nn.Linear(2 * edge_dim, edge_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_state, edge_mask):
        b, k, _ = x.shape

        qkv = self.qkv(self.node_norm1(x)).view(b, k, 3, self.num_heads, self.head_dim)
        q, key, value = qkv.unbind(dim=2)

        logits = torch.einsum("bihd,bjhd->bhij", q, key) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge_state).permute(0, 3, 1, 2)

        attn = torch.softmax(logits, dim=-1)

        edge_value = self.edge_value(edge_state).view(b, k, k, self.num_heads, self.head_dim)
        value_j = value[:, None, :, :, :]
        messages = value_j + edge_value

        attn_msg = attn.permute(0, 2, 3, 1).unsqueeze(-1)
        message = (attn_msg * messages).sum(dim=2).reshape(b, k, self.d_model)

        x = x + self.dropout(self.out_proj(message))
        x = x + self.dropout(self.ffn(self.node_norm2(x)))

        hi = x[:, :, None, :].expand(-1, -1, k, -1)
        hj = x[:, None, :, :].expand(-1, k, -1, -1)

        edge_input = torch.cat([hi, hj, self.edge_norm(edge_state)], dim=-1)
        edge_state = edge_state + self.dropout(self.edge_update(edge_input))
        edge_state = edge_state * edge_mask[..., None]

        return x, edge_state


class CSIRobustGraphTransformer(nn.Module):
    def __init__(
        self,
        num_rx=256,
        num_users=16,
        d_model=128,
        num_heads=8,
        num_layers=8,
        edge_dim=32,
        ffn_dim=256,
        dropout=0.05,
        correction_scale=1.0,
    ):
        super().__init__()

        self.num_rx = int(num_rx)
        self.num_users = int(num_users)
        self.d_model = int(d_model)
        self.correction_scale = float(correction_scale)

        self.channel_encoder = nn.Sequential(
            nn.Linear(2 * self.num_rx, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        self.layers = nn.ModuleList([
            EdgeAwareGraphTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                edge_dim=edge_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.channel_residual_head = nn.Linear(d_model, 2 * self.num_rx)

        nn.init.zeros_(self.channel_residual_head.weight)
        nn.init.zeros_(self.channel_residual_head.bias)

        mask = torch.ones(self.num_users, self.num_users)
        mask.fill_diagonal_(0.0)
        self.register_buffer("edge_mask", mask[None])

    @staticmethod
    def sufficient_statistics(y_white, h_white):
        z = torch.einsum("bmk,bm->bk", h_white.conj(), y_white)
        gram = torch.einsum("bmk,bml->bkl", h_white.conj(), h_white)
        return z, gram

    def _graph_features(self, y_white, h_white):
        z, gram = self.sufficient_statistics(y_white, h_white)

        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
        denom = torch.sqrt(diag[:, :, None] * diag[:, None, :]).clamp_min(1e-8)
        rho = gram / denom

        mask = self.edge_mask.to(rho.real.dtype)
        abs_rho = rho.abs() * mask

        mf = z / diag
        mean_coupling = abs_rho.sum(dim=-1) / float(self.num_users - 1)
        max_coupling = abs_rho.max(dim=-1).values
        rms_coupling = torch.sqrt(abs_rho.square().sum(dim=-1) / float(self.num_users - 1))

        h_user = h_white.transpose(1, 2)
        h_power = h_user.abs().square().mean(dim=-1).clamp_min(1e-8)
        h_scale = torch.sqrt(h_power)[..., None]
        h_unit = h_user / h_scale

        channel_features = torch.cat([h_unit.real, h_unit.imag], dim=-1)

        scalar_features = torch.stack([
            mf.real,
            mf.imag,
            torch.log1p(diag),
            torch.log1p(h_power),
            mean_coupling,
            max_coupling,
            rms_coupling,
        ], dim=-1)

        edge_raw = torch.stack([
            rho.real,
            rho.imag,
            rho.abs(),
            rho.abs().square(),
        ], dim=-1)

        edge_raw = edge_raw * self.edge_mask[..., None]

        return z, gram, h_scale, channel_features, scalar_features, edge_raw

    def forward(self, y_white, h_white):
        if y_white.ndim != 2:
            raise ValueError(f"Expected y_white [B,M], got {tuple(y_white.shape)}")

        if h_white.ndim != 3:
            raise ValueError(f"Expected h_white [B,M,K], got {tuple(h_white.shape)}")

        if h_white.shape[1] != self.num_rx:
            raise ValueError(f"Expected {self.num_rx} Rx antennas, got {h_white.shape[1]}")

        if h_white.shape[2] != self.num_users:
            raise ValueError(f"Expected {self.num_users} users, got {h_white.shape[2]}")

        z_input, gram_input, h_scale, channel_features, scalar_features, edge_raw = self._graph_features(y_white, h_white)

        node_state = self.channel_encoder(channel_features) + self.scalar_encoder(scalar_features)
        edge_state = self.edge_encoder(edge_raw) * self.edge_mask[..., None]

        for layer in self.layers:
            node_state, edge_state = layer(node_state, edge_state, self.edge_mask)

        node_state = self.final_norm(node_state)

        residual_raw = self.channel_residual_head(node_state)
        residual_real, residual_imag = residual_raw.chunk(2, dim=-1)

        residual_normalized = torch.complex(torch.tanh(residual_real), torch.tanh(residual_imag)) / math.sqrt(2.0)
        delta_h_user = self.correction_scale * h_scale * residual_normalized
        delta_h = delta_h_user.transpose(1, 2)

        h_refined = h_white + delta_h
        z_refined, gram_refined = self.sufficient_statistics(y_white, h_refined)

        delta_power = delta_h.abs().square().mean(dim=(1, 2))
        input_power = h_white.abs().square().mean(dim=(1, 2)).clamp_min(1e-8)

        return {
            "h_input": h_white,
            "h_refined": h_refined,
            "delta_h": delta_h,
            "z_input": z_input,
            "gram_input": gram_input,
            "z_refined": z_refined,
            "gram_refined": gram_refined,
            "relative_delta_rms": torch.sqrt(delta_power / input_power),
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CSIRobustGraphTransformer().to(device)

    y = torch.randn(32, 256, device=device) + 1j * torch.randn(32, 256, device=device)
    h = torch.randn(32, 256, 16, device=device) + 1j * torch.randn(32, 256, 16, device=device)

    y = y.to(torch.complex64)
    h = h.to(torch.complex64)

    with torch.no_grad():
        out = model(y, h)

    print("y              :", tuple(y.shape))
    print("h input        :", tuple(h.shape))
    print("h refined      :", tuple(out["h_refined"].shape))
    print("z              :", tuple(out["z_refined"].shape))
    print("Gram           :", tuple(out["gram_refined"].shape))
    print("Initial max ΔH :", (out["h_refined"] - h).abs().max().item())
    print("Parameters     :", sum(p.numel() for p in model.parameters()))