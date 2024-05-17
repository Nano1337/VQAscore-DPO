import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .base_orpo_trainer import ORPOTrainer

class LlavaORPOTrainer(ORPOTrainer):
        
    def concatenated_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images = inputs["images"]
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        reject_input_ids = inputs["reject_input_ids"]
        reject_labels = inputs["reject_labels"]
        reject_attention_mask = inputs["reject_attention_mask"]
            
        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])
        batch_input_ids = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_input_ids.dtype, device=chosen_input_ids.device)
        batch_labels = torch.ones((chosen_input_ids.shape[0]*2, max_dim), dtype=chosen_labels.dtype, device=chosen_labels.device) * -100
        batch_attention_mask = torch.zeros((chosen_input_ids.shape[0]*2, max_dim), device=chosen_attention_mask.device).to(torch.bool)
        batch_input_ids[:chosen_input_ids.shape[0], :chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[reject_input_ids.shape[0]:, :reject_input_ids.shape[1]] = reject_input_ids
        batch_labels[:chosen_labels.shape[0], :chosen_labels.shape[1]] = chosen_labels
        batch_labels[reject_labels.shape[0]:, :reject_labels.shape[1]] = reject_labels
        batch_attention_mask[:chosen_attention_mask.shape[0], :chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[reject_attention_mask.shape[0]:, :reject_attention_mask.shape[1]] = reject_attention_mask

        # prepare inputs
        multimodal_preprocess_kwargs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": torch.cat([images, images], dim=0),
            "labels": batch_labels,
            "position_ids": None,
            "past_key_values": None,
        }
        (
            batch_input_ids,
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,
            batch_labels
        ) = self.model.prepare_inputs_labels_for_multimodal(
            **multimodal_preprocess_kwargs
        )
        
        # calculate logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.is_encoder_decoder: 
            labels = batch_labels.clone()
        else: 
            labels = batch_labels.clone() 
            attention_mask = batch_attention_mask
            labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)
        
        len_chosen = chosen_input_ids.shape[0]
        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps( # FIXME: batch_labels memory access errors here
            all_logits,
            batch_labels, # must clone else there's memory access issues
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder, 
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # don't count image embeds logits
        loss_mask = batch_labels != -100 
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]

        # don't know if this is necessary? 
        chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        chosen_logits = sum(chosen_logits)/len_chosen
        rejected_logits = sum(rejected_logits)/len_chosen

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)
