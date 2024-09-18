
from transformers import LlamaForCausalLM, LlamaTokenizer

# import static cache 


import torch

class LlamaForMLM(LlamaForCausalLM):

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            num_logits_to_keep=None,
            **kwargs,
        ):
        
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        current_attention_mask = model_inputs.get("attention_mask")
        ones_like = torch.ones_like(current_attention_mask)
        model_inputs.update(
                {
                    "attention_mask": ones_like,
                }
            )