import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn import Linear4bit


class OrthogonalSVDAdapter(nn.Module):
    """
    Fine-tuning adapter that adds updates strictly in the orthogonal complement 
    of the pre-trained weight matrix's row and column spaces.
    """

    def __init__(self, base_linear: nn.Module, k: int, generator: torch.Generator):
        super().__init__()

        self.base_layer = base_linear
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # One-time dequantize for SVD
        with torch.no_grad():
            W0 = dequantize_4bit(
                base_linear.weight.data, base_linear.weight.quant_state
            ).to(torch.float32)

        # m, n = W0.shape
        # print(f"W0 shape: {m} x {n}, dtype: {W0.dtype}")
        # print(f"NaN count: {torch.isnan(W0).sum().item()}")
        # print(f"Inf count: {torch.isinf(W0).sum().item()}")
        # print(f"Value range: [{W0.min().item():.4f}, {W0.max().item():.4f}]")

        # U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        U, S, Vh = torch.linalg.svd(W0, full_matrices=True)
        V = Vh.T

        # =====
        # tol = S.max() * max(m, n) * torch.finfo(S.dtype).eps
        # r_eff = int((S > tol).sum().item())
        # print(f"{m = }, {n = } and {r_eff = }")
        # max_k = min(m - r_eff, n - r_eff)
        # if k > max_k:
        #     raise ValueError("No orthogonal complement available. Lower k or use LoRA.")

        # Ru = torch.randn(m, k, device=generator.device, generator=generator)
        # Pu = Ru - U @ (U.T @ Ru)  # Project onto orthogonal complement of U
        # Qu, _ = torch.linalg.qr(Pu, mode='reduced')
        # Rv = torch.randn(n, k, device=generator.device, generator=generator)
        # Pv = Rv - Vh.T @ (Vh @ Rv)  # Project onto orthogonal complement of V
        # Qv, _ = torch.linalg.qr(Pv, mode='reduced')

        # self.register_buffer('U_perp', Qu)
        # self.register_buffer('V_perp', Qv)
        # self.alpha = nn.Parameter(torch.zeros(k))
        # =====

        # tol = S.max() * max(m, n) * torch.finfo(S.dtype).eps
        # r_eff = int((S > tol).sum().item())
        # print(f"{m = }, {n = } and {r_eff = }")
        # max_k = min(m - r_eff, n - r_eff)
        # k = min(k, max_k)
        # if k <= 0:
        #     raise ValueError("No orthogonal complement available. Lower k or use LoRA.")
        print(f"1-st svdval: {S[0].item():.4f} and k-th last svdval: {S[-k].item():.4f}")

        # r_eff:(r_eff + k)
        self.register_buffer("U_perp", U[:, -k:])
        self.register_buffer("V_perp", V[:, -k:])
        self.alpha = nn.Parameter(torch.zeros(k))

        del W0, U, V, S
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        delta = (x @ (self.V_perp * self.alpha)) @ self.U_perp.T
        return base_out + delta

    def alt_init(self, linear_layer: nn.Linear, k: int, seed: int = 42):
        super().__init__()

        quant_weight = linear_layer.weight
        self.weight_data = quant_weight.data
        self.quant_state = quant_weight.quant_state
        with torch.no_grad():
            W_0 = dequantize_4bit(self.weight_data, self.quant_state)

        self.bias = linear_layer.bias.detach() if linear_layer.bias is not None else None

        m, n = W_0.shape

        # U: m x r, V: n x r, where r = min(m, n)
        U, S, Vh = torch.linalg.svd(W_0, full_matrices=False)
        r = U.shape[1]

        # Maximum possible k is bounded by the dimension of the orthogonal complements
        max_k = min(m - r, n - r)
        if k > max_k:
            raise ValueError(
                f"k={k} exceeds maximum allowable value {max_k} for "
                f"orthogonal complements of a {m}x{n} matrix with rank {r}."
            )
        else:
            print(f"Debug: {max_k = }")

        # Use a local generator to avoid polluting global RNG state
        generator = torch.Generator(device=self.original_weight.device)
        generator.manual_seed(seed)

        # Sample & orthonormalize U_perp (m × k)
        Ru = torch.randn(m, k, device=self.original_weight.device, generator=generator)
        Pu = Ru - U @ (U.T @ Ru)  # Project onto orthogonal complement of U
        Qu, _ = torch.linalg.qr(Pu, mode='reduced')

        # Sample & orthonormalize V_perp (n × k)
        Rv = torch.randn(n, k, device=self.original_weight.device, generator=generator)
        Pv = Rv - Vh.T @ (Vh @ Rv)  # Project onto orthogonal complement of V
        Qv, _ = torch.linalg.qr(Pv, mode='reduced')

        self.register_buffer('U_perp', Qu)
        self.register_buffer('V_perp', Qv)

        # # Effective rank calculation
        # tol = S.max() * max(m, n) * torch.finfo(S.dtype).eps
        # r_eff = int(torch.sum(S > tol).item())
        # max_k = min(m - r_eff, n - r_eff)
        # if k > max_k:
        #     print(f"Warning: k={k} > max_k={max_k}. Clipping to max_k.")
        #     k = max_k
        # if k == 0:
        #     raise ValueError("Matrix is numerically full-rank. No orthogonal complement available. "
        #                      "Try lowering rank_tol_ratio or using standard LoRA.")
            
        # # Use trailing columns of full SVD as exact orthogonal complements
        # # U_perp: m x k, V_perp: n x k
        # self.register_buffer('U_perp', U[:, r_eff:r_eff + k])
        # self.register_buffer('V_perp', V[:, r_eff:r_eff + k])

        # Learnable coefficients (initialized to zero to preserve pre-trained behavior at start)
        self.alpha = nn.Parameter(torch.zeros(k))

    def alt_forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            W_0 = dequantize_4bit(self.weight_data, self.quant_state)

            out = F.linear(x, W_0, self.bias)

        delta = (x @ (self.V_perp * self.alpha)) @ self.U_perp.T

        return out + delta


def apply_orthogonal_adapters(model, target_regex=r"layers\.\d+\.(mlp|self_attn)\..+_proj", k=8):
    pattern = re.compile(target_regex)
    generator = torch.Generator(device=model.device)

    for name, module in model.named_modules():
        # if pattern.search(name) and isinstance(module, (nn.Linear, Linear4bit)):
        if pattern.search(name) and isinstance(module, Linear4bit):
            print(f"Applying orthogonal adapter to {name} with ", end="")
            adapter = OrthogonalSVDAdapter(module, k, generator)

            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module, child_name = model, name

            setattr(parent_module, child_name, adapter)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            param.requires_grad = "alpha" in name
        elif "alpha" in name:
            print(name, "does not require grad")

    return model

