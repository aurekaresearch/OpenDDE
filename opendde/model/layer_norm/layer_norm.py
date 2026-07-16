# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2024 ByteDance and/or its affiliates.
#
# Copyright 2021- HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import logging
import numbers
import os
import threading
from typing import Any, Optional, Union, cast

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

_FAST_LAYER_NORM_REQUIRED_SYMBOLS = (
    "forward_none_affine",
    "forward_with_bias_affine",
    "forward_with_weight_affine",
    "forward_with_both_affine",
    "backward_none_affine",
    "backward_with_bias_affine",
    "backward_with_weight_affine",
    "backward_with_both_affine",
)

fast_layer_norm_cuda_v2 = None
_fast_layer_norm_load_attempted = False
_fast_layer_norm_load_lock = threading.Lock()


def _validate_fast_layer_norm_extension(extension: Any) -> Any:
    missing_symbols = [
        symbol
        for symbol in _FAST_LAYER_NORM_REQUIRED_SYMBOLS
        if not callable(getattr(extension, symbol, None))
    ]
    if missing_symbols:
        raise ImportError(
            "Fast LayerNorm extension is missing required symbols: "
            + ", ".join(missing_symbols)
        )
    return extension


def _load_fast_layer_norm_cuda_v2() -> Any:
    """Load or compile the optional extension on its first CUDA use."""
    global fast_layer_norm_cuda_v2
    global _fast_layer_norm_load_attempted

    if not torch.cuda.is_available():
        return None
    if _fast_layer_norm_load_attempted:
        return fast_layer_norm_cuda_v2

    with _fast_layer_norm_load_lock:
        if _fast_layer_norm_load_attempted:
            return fast_layer_norm_cuda_v2

        try:
            extension = importlib.import_module(
                ".fast_layer_norm_cuda_v2",
                package=__package__,
            )
            extension = _validate_fast_layer_norm_extension(extension)
        except Exception:
            try:
                from opendde.model.layer_norm.torch_ext_compile import compile

                current_dir = os.path.dirname(__file__)
                extension = compile(
                    name="fast_layer_norm_cuda_v2",
                    sources=[
                        os.path.join(current_dir, "kernel", file)
                        for file in (
                            "layer_norm_cuda.cpp",
                            "layer_norm_cuda_kernel.cu",
                        )
                    ],
                    extra_include_paths=[os.path.join(current_dir, "kernel")],
                )
                extension = _validate_fast_layer_norm_extension(extension)
            except Exception as error:
                logger.warning(
                    "Fast LayerNorm CUDA extension is unavailable; using torch "
                    "layer_norm instead: %s",
                    error,
                )
                extension = None

        fast_layer_norm_cuda_v2 = extension
        _fast_layer_norm_load_attempted = True
        return extension


class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        normalized_shape: torch.Size,
        eps: float,
    ) -> torch.Tensor:
        d = input.dtype
        fast_layer_norm = cast(Any, fast_layer_norm_cuda_v2)

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()

        if weight is None:
            if bias is None:
                output, mean, invvar = fast_layer_norm.forward_none_affine(
                    input_, ctx.normalized_shape, ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm.forward_with_bias_affine(
                    input_, ctx.normalized_shape, bias.to(d), ctx.eps
                )
        else:
            if bias is None:
                (
                    output,
                    mean,
                    invvar,
                ) = fast_layer_norm.forward_with_weight_affine(
                    input_, ctx.normalized_shape, weight.to(d), ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm.forward_with_both_affine(
                    input_,
                    ctx.normalized_shape,
                    weight.to(d),
                    bias.to(d),
                    ctx.eps,
                )
        ctx.save_for_backward(input_, weight, bias, mean, invvar)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Optional[torch.Tensor], ...]:
        (grad_output,) = grad_outputs
        grad_output = cast(torch.Tensor, grad_output)
        d = grad_output.dtype
        fast_layer_norm = cast(Any, fast_layer_norm_cuda_v2)
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if weight_ is None:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm.backward_none_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm.backward_with_bias_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        else:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm.backward_with_weight_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm.backward_with_both_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        return (
            grad_input,
            None if weight_ is None else grad_weight,
            None if bias_ is None else grad_bias,
            None,
            None,
        )


class FusedLayerNorm(torch.nn.Module):
    """
    Args:
        normalized_shape (int or list or torch.Size) input shape from an expected input of size
        create_scale (bool) If set to False, the layer will not learn an additive weight, Default: True
        create_offset (bool) If set to False, the layer will not learn an additive bias, Default: True
        eps (float) a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        normalized_shape: Union[int, list[int], torch.Size],
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super(FusedLayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape_tuple = (int(normalized_shape),)
        else:
            normalized_shape_tuple = tuple(
                cast(Union[list[int], torch.Size], normalized_shape)
            )
        self.normalized_shape = torch.Size(normalized_shape_tuple)
        self.eps = eps
        if create_scale:
            self.weight = Parameter(torch.ones(*self.normalized_shape))
        else:
            self.weight = None

        if create_offset:
            self.bias = Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda or _load_fast_layer_norm_cuda_v2() is None:
            return F.layer_norm(
                input,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        return FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps
        )
