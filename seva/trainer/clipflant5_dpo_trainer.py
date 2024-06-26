import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .base_dpo_trainer import BaseDPOTrainer

class CLIPFlanT5DPOTrainer(BaseDPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: this setting is VERY IMPORTANT for T5 architectures, an encoder-decoder
        self.is_encoder_decoder = True
        
    def concatenated_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        images = inputs["images"]
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_decoder_attention_mask = inputs["chosen_decoder_attention_mask"]
        reject_input_ids = inputs["reject_input_ids"]
        reject_labels = inputs["reject_labels"]
        reject_attention_mask = inputs["reject_attention_mask"]
        reject_decoder_attention_mask = inputs["reject_decoder_attention_mask"]
    
        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])
        batch_input_ids = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_input_ids.dtype, device=chosen_input_ids.device)
        batch_labels = torch.ones((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        batch_attention_mask = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), device=chosen_attention_mask.device).to(torch.bool)
        batch_decoder_attention_mask = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), device=chosen_attention_mask.device).to(torch.bool)
        batch_input_ids[:chosen_input_ids.shape[0], :chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[reject_input_ids.shape[0]:, :reject_input_ids.shape[1]] = reject_input_ids
        batch_labels[:chosen_labels.shape[0], :chosen_labels.shape[1]] = chosen_labels
        batch_labels[reject_labels.shape[0]:, :reject_labels.shape[1]] = reject_labels
        batch_attention_mask[:chosen_attention_mask.shape[0], :chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[reject_attention_mask.shape[0]:, :reject_attention_mask.shape[1]] = reject_attention_mask
        batch_decoder_attention_mask[:chosen_decoder_attention_mask.shape[0], :chosen_decoder_attention_mask.shape[1]] = chosen_decoder_attention_mask
        batch_decoder_attention_mask[reject_decoder_attention_mask.shape[0]:, :reject_decoder_attention_mask.shape[1]] = reject_decoder_attention_mask

        model_forward_kwargs = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'decoder_attention_mask': batch_decoder_attention_mask,
            'labels': batch_labels,
            'images': torch.cat([images, images], dim=0),
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }
        # calculate logits
        all_logits = model.forward(
            **model_forward_kwargs
        ).logits.to(torch.float32) # Shape (B*2, S, H), H=32128

        cal_batch_logp = self._get_batch_logps
        all_logps = cal_batch_logp(
            all_logits,
            batch_labels,
            average_log_prob=False,
        )

        len_chosen = chosen_input_ids.shape[0]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # don't count image embeds logits
        loss_mask = batch_labels != 0 # self._get_batch_logps already modified to zero out -100 pad id
        # logits = all_logits[loss_mask]

        """
        All logits shape is (B*2, S, H)
        Loss mask shape is (B*2, S)
        Logits after applying loss mask is [(2, H)]*B*2 where 2 is the response and stop token
        TODO: can modify logic below to handle parallelization fairly easily
        """

        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]

        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]

        # print("Chosen logits array", chosen_logits)
        # print("Rejected logits array", rejected_logits)
        # exit()

        chosen_logits = [l.detach().cpu() for l in chosen_logits]
        rejected_logits = [l.detach().cpu() for l in rejected_logits]

        # instead of averaging across all logits, find argmax of each tensor then average those logits
        assert len(chosen_logits) == len(rejected_logits)
        for i in range(len(chosen_logits)): 
            chosen_logits[i] = torch.mean(torch.max(chosen_logits[i], dim=-1).values)
            rejected_logits[i] = torch.mean(torch.max(rejected_logits[i], dim=-1).values)

        # print("Chose logits argmax", chosen_logits)
        # print("Rejected logits argmax", rejected_logits)

        chosen_logits = sum(chosen_logits)/len_chosen
        rejected_logits = sum(rejected_logits)/len_chosen

        # print("Chosen logits avg", chosen_logits)
        # print("Rejected logits avg", rejected_logits)
        # exit()

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(self.model, inputs)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, inputs)

        policy_rejected_logps = policy_rejected_logps
        reference_rejected_logps = reference_rejected_logps
           
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits

        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            
        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
