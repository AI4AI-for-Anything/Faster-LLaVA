#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import os

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Optional Apex RMSNorm fast path; enabled only for an explicit benchmark candidate.
if os.environ.get("LLAVA_APEX_RMSNORM") == "1":
    from apex.normalization import FusedRMSNorm
    import transformers.models.llama.modeling_llama as _llama_impl
    _llama_impl.LlamaRMSNorm = FusedRMSNorm
if os.environ.get("LLAVA_BF16_REDUCED_REDUCTION") == "1":
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
if os.environ.get("LLAVA_UNPAD_MLP") == "1":
    from ..unpad_mlp import enable_unpadded_mlp
    enable_unpadded_mlp()
if os.environ.get("LLAVA_UNPAD_ATTN_PROJ") == "1":
    from ..unpad_attention import enable_unpadded_projections
    enable_unpadded_projections()
if os.environ.get("LLAVA_PACKED_ATTN") == "1":
    from ..packed_attention import enable_packed_attention
    enable_packed_attention()
if os.environ.get("LLAVA_LIGER_SWIGLU") == "1":
    import random as _random
    import triton.runtime.cache as _triton_cache
    # Triton 2.1 draws temporary cache-file suffixes from Python's global RNG.
    # Keep filesystem naming independent of the training RNG, even on a cold
    # compilation. Only the cache module's local reference is replaced.
    _triton_cache.random = _random.SystemRandom()
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from transformers.models.llama.modeling_llama import LlamaMLP
    _original_mlp_forward = LlamaMLP.forward

    def _fused_mlp_forward(self, x):
        if self.config.pretraining_tp > 1:
            return _original_mlp_forward(self, x)
        return self.down_proj(LigerSiLUMulFunction.apply(
            self.gate_proj(x), self.up_proj(x)))

    LlamaMLP.forward = _fused_mlp_forward

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self._use_fused_head_loss = os.environ.get("LLAVA_FUSED_HEAD_LOSS") == "1"
        if self._use_fused_head_loss:
            if self.config.pretraining_tp != 1:
                raise ValueError("Fused head loss requires pretraining_tp=1")
            from ..fused_head_loss import enable_fused_head
            enable_fused_head(self.lm_head)
        if os.environ.get("LLAVA_PACKED_DECODER") == "1":
            from ..packed_decoder import enable_packed_decoder
            enable_packed_decoder(self.model)
        if os.environ.get("LLAVA_RECOMPUTE_SWIGLU") == "1":
            from ..recompute_swiglu import enable_recomputed_swiglu
            enable_recomputed_swiglu(self.model)
        if os.environ.get("LLAVA_RMS_MEMORY_EFFICIENT") == "1":
            from apex.normalization import FusedRMSNorm
            norms = [module for module in self.model.modules()
                     if isinstance(module, FusedRMSNorm)]
            if not norms:
                raise ValueError("Memory-efficient RMSNorm requires the Apex RMSNorm path")
            for norm in norms:
                norm.memory_efficient = True
            print(f"Enabled memory-efficient Apex RMSNorm for {len(norms)} modules", flush=True)

    def get_model(self):
        return self.model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        keep = int(os.environ.get("LLAVA_KEEP_ACTIVATION_LAYERS", "0"))
        if keep:
            layers = self.model.layers
            if not 0 < keep <= len(layers):
                raise ValueError("LLAVA_KEEP_ACTIVATION_LAYERS exceeds decoder depth")
            retained = frozenset(layers[-keep:])
            checkpoint = self.model._gradient_checkpointing_func

            def selective_checkpoint(function, *args, **kwargs):
                if getattr(function, "__self__", None) in retained:
                    return function(*args, **kwargs)
                return checkpoint(function, *args, **kwargs)

            self.model._gradient_checkpointing_func = selective_checkpoint

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        if self._use_fused_head_loss and labels is not None:
            from ..fused_head_loss import loss_only_forward
            return loss_only_forward(
                self, labels=labels, return_dict=return_dict,
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
