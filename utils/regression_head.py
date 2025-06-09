import torch
import torch.nn as nn
from adapters.heads.base import PredictionHead
from transformers.modeling_outputs import CausalLMOutput


class CustomRegressionHead(PredictionHead):

    def __init__(self, model, head_name, **kwargs):
        super().__init__(head_name)
        hidden_size = model.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1) 

    def forward(self,
                outputs,
                cls_output=None,
                attention_mask=None,
                return_dict=False,
                **kwargs):

        pooled_output = outputs.pooler_output if cls_output is None else cls_output
        logits = self.regressor(pooled_output)
        if return_dict:
            return CausalLMOutput(
                loss=None,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return logits
