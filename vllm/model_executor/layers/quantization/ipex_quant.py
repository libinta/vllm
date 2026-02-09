# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch
from packaging import version
from torch.nn import Module

from vllm._ipex_ops import ipex_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
    Fp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    scaled_dequantize,
)  
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform 

MIN_IPEX_VERSION = "2.6.0"


class IPEXConfig(QuantizationConfig):
    """INT8 quantization config class using IPEX for the CPU/XPU backend,
    including AWQ, GPTQ.
    """

    IPEX_QUANT_METHOD_MAP = {
        "awq": 1,
        "gptq": 0,
    }

    def __init__(
        self,
        method: str,
        weight_bits: int,
        group_size: int,
        modules_to_not_convert: list[str] | None = None,
        desc_act: bool | None = None,
        lm_head_quantized: bool | None = None,
        is_sym: bool | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.modules_to_not_convert = modules_to_not_convert or []
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.is_sym = is_sym
        self.pack_factor = 32 // self.weight_bits

        if self.weight_bits not in [4]:
            raise ValueError(
                f"IPEX quantization supports weight bits [4], "
                f"but got {self.weight_bits}."
            )

        if self.method not in ["awq", "gptq"]:
            raise ValueError(
                f"IPEX quantization supports [awq, gptq], but got {self.method}."
            )

    def __repr__(self) -> str:
        return (
            f"IPEXConfig(method={self.method},"
            f"weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "ipex"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IPEXConfig":
        method = cls.get_from_keys(config, ["quant_method"]).lower()
        if method == "awq":
            weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
            is_sym = not cls.get_from_keys_or(config, ["zero_point"], default=False)
            return cls(
                method,
                weight_bits,
                group_size,
                modules_to_not_convert,
                False,
                False,
                is_sym,
            )
        # otherwise for gptq
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        desc_act = cls.get_from_keys_or(config, ["desc_act"], default=False)
        is_sym = cls.get_from_keys_or(config, ["sym"], default=True)
        return cls(
            method, weight_bits, group_size, [], desc_act, lm_head_quantized, is_sym
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        if not current_platform.is_xpu():
            return None

        quant_method = hf_quant_cfg.get("quant_method", "").lower()

        if quant_method in ["awq", "gptq"]:
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.method == "awq":
                if is_layer_skipped(
                    prefix,
                    self.modules_to_not_convert,
                    self.packed_modules_mapping,
                    skip_with_substr=True,
                ):
                    return UnquantizedLinearMethod()
                return IPEXAWQLinearMethod(self)
            if self.method == "gptq":
                return IPEXGPTQLinearMethod(self)
        return None


class IPEXGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ linear method using IPEX for the CPU/XPU backend."""

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method."
            ) from err
        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK

        assert isinstance(self.quant_config, IPEXConfig)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )
        layer.ipex_output_size = layer.qweight.shape[-1]
        g_idx = layer.g_idx if self.quant_config.desc_act else None
        layer.ipex_qlinear = (
            ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
                layer.qweight,
                layer.scales,
                layer.qzeros,
                layer.qweight.size(0),
                layer.ipex_output_size,
                qconfig=qconfig,
                g_idx=g_idx,
                bias=bias,
                group_size=self.quant_config.group_size,
                quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["gptq"],
                weight_qscheme="sym" if self.quant_config.is_sym else "asym",
            )
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))


class IPEXAWQLinearMethod(AWQLinearMethod):
    """AWQ linear method using IPEX for the CPU/XPU backend."""

    def __init__(self, quant_config: IPEXConfig):
        self.quant_config = quant_config  # type: ignore

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer=layer)

        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method."
            ) from err

        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH

        assert isinstance(self.quant_config, IPEXConfig)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=self.quant_config.group_size,
        )

        layer.ipex_output_size = layer.qweight.size(1) * self.quant_config.pack_factor
        layer.ipex_qlinear = (
            ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
                layer.qweight,
                layer.scales,
                layer.qzeros,
                layer.qweight.size(0),
                layer.ipex_output_size,
                qconfig=qconfig,
                bias=bias,
                group_size=self.quant_config.group_size,
                quant_method=IPEXConfig.IPEX_QUANT_METHOD_MAP["awq"],  # type: ignore
                weight_qscheme="sym" if self.quant_config.is_sym else "asym",
            )
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))

class XPUFp8MoEMethodOffline(Fp8MoEMethod):
    def __init__(self, quant_config: "Fp8Config", layer: torch.nn.Module):
        super().__init__(quant_config, layer)
        self.quant_config = quant_config
        # Required attributes
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        fp8_dtype = current_platform.fp8_dtype()
        if not hasattr(layer, "w13_weight_scale"):
            if hasattr(layer, "w13_weight_scale_inv"):
                # Alias the block quant scale name to the expected name
                replace_parameter(layer, "w13_weight_scale", layer.w13_weight_scale_inv)
            else:
                raise AttributeError("Missing w13_weight_scale attribute")

        if not hasattr(layer, "w2_weight_scale"):
            if hasattr(layer, "w2_weight_scale_inv"):
                replace_parameter(layer, "w2_weight_scale", layer.w2_weight_scale_inv)
            else:
                raise AttributeError("Missing w2_weight_scale attribute")

        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

        # Re-initialize w13_scale because we directly quantize
        # merged w13 weights and generate a single scaling factor.
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            layer.local_num_experts,
            dtype=torch.float32,
            device=w13_weight.device,
            ),
            requires_grad=False,
        )
        for expert in range(layer.local_num_experts):
            w13_in = layer.w13_weight.data[expert, :, :]
            w2_in = layer.w2_weight.data[expert, :, :]

            if w13_in.dtype != fp8_dtype:
                # Use proper dequantization utility
                w13_scale = layer.w13_weight_scale[expert] if hasattr(layer, 'w13_weight_scale') else  1.0
                w2_scale = layer.w2_weight_scale[expert] if hasattr(layer, 'w2_weight_scale') else 1.0
                w13_bf16 = scaled_dequantize(w13_in, w13_scale, group_shape=None, out_dtype=torch.bfloat16)
                w2_bf16 = scaled_dequantize(w2_in, w2_scale, group_shape=None, out_dtype=torch.bfloat16)

            w13_weight[expert, :, :], w13_weight_scale[expert] = ( ops.scaled_fp8_quant(w13_bf16))
            w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (ops.scaled_fp8_quant(w2_bf16))
        replace_parameter(layer, "w13_weight", w13_weight)
        replace_parameter(layer, "w2_weight", w2_weight)
        replace_parameter(layer, "w13_weight_scale", w13_weight_scale)

        import intel_extension_for_pytorch as ipex

        ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            w1_scale_inv=layer.w13_weight_scale,
            w2_scale_inv=layer.w2_weight_scale,
            a1_scale_inv=layer.w13_input_scale,
            a2_scale_inv=layer.w2_input_scale,
            use_prepack=True,
            experts_start_id=ep_rank_start,
        )

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        #print("DEBUG: Using monolithic XPU FP8 MoE path")
        return layer.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

class XPUFp8LinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
        
    def convert_e4m3_to_e5m2(self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Convert FP8 E4M3 weights/scales to E5M2 on XPU."""
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Expected e4m3fn, got {weight.dtype}")
         # Use vLLM's robust dequantization with proper group broadcasting
        weight_dequant = scaled_dequantize(
            weight,
            weight_scale,
            group_shape=None,  # Let it infer from scale shape
            out_dtype=torch.bfloat16
        )
        # Reshape to 2D for quantization (merge all but last dimension)
        original_shape = weight_dequant.shape
        weight_dequant_2d = weight_dequant.view(-1, weight_dequant.shape[-1])

        # Requantize to E5M2 using XPU's preferred dtype
        weight_e5m2_2d, scale_e5m2 = ops.scaled_fp8_quant(weight_dequant_2d, scale=None)

        # Reshape back to original dimensions
        weight_e5m2 = weight_e5m2_2d.view(original_shape)
        return weight_e5m2, scale_e5m2, input_scale

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
            # Update the layer with the new values.
            replace_parameter(layer, "weight", qweight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)
            layer.input_scale = None
        else:
            if not hasattr(layer, "weight_scale"):
               if hasattr(layer, "weight_scale_inv"):
                   replace_parameter(layer, "weight_scale", layer.weight_scale_inv)
                   if layer.weight.dtype == torch.float8_e4m3fn and layer.weight.dtype != current_platform.fp8_dtype():
                       weight_e5m2, scale_e5m2, _ = self.convert_e4m3_to_e5m2(
                           layer.weight, layer.weight_scale, getattr(layer, "input_scale", None))
                       replace_parameter(layer, "weight", weight_e5m2)
                       replace_parameter(layer, "weight_scale", scale_e5m2)
               else:
                    raise AttributeError("Missing weight_scale attribute")
            layer.input_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: 
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        output = torch.ops.torch_ipex.fp8_gemm_w8a16(
            x, weight, True, weight_scale, bias
        )
        return output


class XPUFp8MoEMethod(Fp8OnlineMoEMethod):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.local_num_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            replace_parameter(layer, "w13_weight", w13_weight)
            replace_parameter(layer, "w2_weight", w2_weight)

        import intel_extension_for_pytorch as ipex

        ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            w1_scale_inv=layer.w13_weight_scale,
            w2_scale_inv=layer.w2_weight_scale,
            a1_scale_inv=layer.w13_input_scale,
            a2_scale_inv=layer.w2_input_scale,
            use_prepack=True,
            experts_start_id=ep_rank_start,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return layer.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
        )
