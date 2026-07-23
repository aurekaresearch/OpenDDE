# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2021 AlQuraishi Laboratory

import contextlib
import math
import os
import sys
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from opendde.model.utils import (
    chunk_layer,
    flatten_final_dims,
    is_fp16_enabled,
    permute_final_dims,
)

_use_fast_layer_norm = os.getenv("LAYERNORM_TYPE", "torch") == "fast_layernorm"
if _use_fast_layer_norm:
    try:
        from opendde.model.layer_norm.layer_norm import FusedLayerNorm
    except (ImportError, OSError, RuntimeError):
        _use_fast_layer_norm = False


def _prod(nums: Union[List[int], Tuple[int, ...], torch.Size]) -> int:
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(
    linear_weight_shape: Union[torch.Size, Tuple[int, ...]], fan: str = "fan_in"
) -> Union[int, float]:
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(
    weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in"
) -> None:
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor) -> None:
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


@contextlib.contextmanager
def skip_random_init():
    """
    Skips random weight initialization.

    Skips the following initialization work:
    * trunc_normal_init_ in opendde.model.triangular.layers
    * trunc_normal_init_ in opendde.model.modules.primitives
    * reset_parameters in torch.nn.Linear which includes kaiminig_uniform_
    """
    def _noop_init(*_args, **_kwargs):
        pass

    from opendde.model.modules import primitives

    originals = [
        (sys.modules[__name__], "trunc_normal_init_"),
        (primitives, "trunc_normal_init_"),
        (torch.nn.Linear, "reset_parameters"),
    ]
    saved = [(obj, attr, getattr(obj, attr)) for obj, attr in originals]
    try:
        for obj, attr in originals:
            setattr(obj, attr, _noop_init)
        yield
    finally:
        for obj, attr, value in saved:
            setattr(obj, attr, value)


class OpenfoldLinear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.

    Args:
        in_dim:
            The final dimension of inputs to the layer
        out_dim:
            The final dimension of layer outputs
        bias:
            Whether to learn an additive bias. True by default
        init:
            The initializer to use. Choose from:

            "default": LeCun fan-in truncated normal initialization
            "relu": He initialization w/ truncated normal distribution
            "glorot": Fan-average Glorot uniform initialization
            "gating": Weights=0, Bias=1
            "normal": Normal initialization with std=1/sqrt(fan_in)
            "final": Weights=0, Bias=0

            Overridden by init_fn if the latter is not None.
        init_fn:
            A custom initializer taking weight and bias as inputs.
            Overrides init if not None.
        precision:
            Data type for high precision calculation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[
            Callable[[torch.Tensor, Optional[torch.Tensor]], None]
        ] = None,
        precision: Optional[torch.dtype] = None,
    ):
        super(OpenfoldLinear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        if self.precision is not None:
            with torch.amp.autocast("cuda", enabled=False):
                precision_input = input.to(dtype=self.precision)
                precision_weight = self.weight.to(dtype=self.precision)
                bias = (
                    self.bias.to(dtype=self.precision)
                    if self.bias is not None
                    else None
                )
                return nn.functional.linear(
                    precision_input,
                    precision_weight,
                    bias,
                ).to(dtype=d)

        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return nn.functional.linear(input, self.weight, self.bias)


class OpenFoldLayerNorm(nn.Module):
    def __init__(
        self,
        c_in: int,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super(OpenFoldLayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps

        if self.create_scale:
            self.weight = nn.Parameter(torch.ones(c_in))
        else:
            self.register_parameter("weight", None)
        if self.create_offset:
            self.bias = nn.Parameter(torch.zeros(c_in))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d) if self.weight is not None else None,
                    self.bias.to(dtype=d) if self.bias is not None else None,
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )
        return out


def LayerNorm(
    c_in: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-5,
) -> nn.Module:
    if _use_fast_layer_norm:
        return FusedLayerNorm(
            c_in, create_scale=create_scale, create_offset=create_offset, eps=eps
        )
    return OpenFoldLayerNorm(c_in, create_scale, create_offset, eps)


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.

    Args:
        c_q:
            Input dimension of query data
        c_k:
            Input dimension of key data
        c_v:
            Input dimension of value data
        c_hidden:
            Per-head hidden dimension
        no_heads:
            Number of attention heads
        gating:
            Whether the output should be gated using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ) -> None:
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = OpenfoldLinear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = OpenfoldLinear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = OpenfoldLinear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = OpenfoldLinear(
            self.c_hidden * self.no_heads, self.c_q, bias=False, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = OpenfoldLinear(
                self.c_q, self.c_hidden * self.no_heads, bias=False, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        triangle_attention: str = "torch",
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": NVIDIA cuEquivariance attention kernel
        Returns
            [*, Q, C_q] attention update
        """
        if triangle_attention not in {"torch", "cuequivariance"}:
            raise ValueError(
                "Invalid triangle_attention. Options: 'cuequivariance', 'torch'."
            )

        if biases is None:
            biases = []

        use_cuequivariance = (
            triangle_attention == "cuequivariance"
            and q_x.is_cuda
            and q_x.shape[-2] > 16
        )
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=not use_cuequivariance)

        if use_cuequivariance:
            # Notes:
            #     (1) Context is saved for backward pass. You don't need to save it manually.
            #     (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32,
            #         set it from torch global scope
            #     (3) Triangle attention kernel supports: all hidden_dim<=32 and divisible by 4 for tf32/fp32,
            #         and for all hidden_dim<=128 and divisible by 8 for bf16/fp16. In the rare instance that
            #         the kernel does not support an input config, fallback to torch is enabled instead of erroring out.
            scale = 1.0 / math.sqrt(self.c_hidden)
            o = cuequivariance_triangular_attn(
                q, k, v, biases[1].float(), (biases[0] == 0).bool(), scale
            )[0]
            o = o.transpose(-2, -3)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


def cuequivariance_triangular_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    from cuequivariance_torch.primitives.triangle import triangle_attention

    return triangle_attention(q, k, v, bias, mask=mask, scale=scale)


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.

    Args:
        c_m:
            MSA embedding channel dimension
        c_z:
            Pair embedding channel dimension
        c_hidden:
            Hidden channel dimension
        eps:
            Small constant for numerical stability. Defaults to 1e-3.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3) -> None:
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = OpenfoldLinear(c_m, c_hidden, bias=False)
        self.linear_2 = OpenfoldLinear(c_m, c_hidden, bias=False)
        self.linear_out = OpenfoldLinear(c_hidden**2, c_z, init="final")

    def _opm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def _forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm

        return outer

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward(m.float(), mask, chunk_size, inplace_safe)
        else:
            return self._forward(m, mask, chunk_size, inplace_safe)
