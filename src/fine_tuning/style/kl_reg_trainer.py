import torch
import torch.nn.functional as F
from transformers import Trainer


class KLRegularizedTrainer(Trainer):
    def __init__(self, base_model, kl_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.base_model.eval()
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Ensure base model matches device
        if self.base_model.device != model.device:
            self.base_model = self.base_model.to(model.device)

        # Extract sample type flag
        # is_general_list = inputs.pop("is_general", None)
        # if is_general_list is not None:
        #     is_general = torch.tensor(is_general_list, device=model.device, dtype=torch.bool)
        # else:
        #     is_general = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=model.device)
        is_general_list = inputs.get("is_general")
        if is_general_list is not None:
            if isinstance(is_general_list, list):
                is_general = torch.tensor(is_general_list, device=model.device, dtype=torch.bool)
            else:
                is_general = is_general_list.to(model.device, dtype=torch.bool)
        else:
            is_general = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=model.device)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is not None:
            # Shift for causal LM (predict next token)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_is_general = is_general.unsqueeze(1).expand_as(shift_labels)

            # Mask for valid tokens (ignore padding)
            valid_mask = (shift_labels != -100).float()

            # Standard Cross-Entropy Loss for TRUE training samples
            lm_loss = torch.nn.CrossEntropyLoss(reduction='none')
            ce_loss_flat = lm_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ce_loss = ce_loss_flat.view(shift_labels.size())
            ce_mask = valid_mask * (~shift_is_general).float()
            ce_loss = (ce_loss * ce_mask).sum() / ce_mask.sum().clamp(min=1e-9)

            # KL Divergence Loss for GENERAL purpose samples
            kl_loss = torch.tensor(0.0, device=logits.device)
            if torch.any(shift_is_general):
                with torch.no_grad():
                    base_outputs = self.base_model(**inputs)
                    base_logits = base_outputs.logits
                    shift_base_logits = base_logits[:, :-1, :].contiguous()

                # KL(P_base || P_adapted)
                p_log_probs = F.log_softmax(shift_base_logits, dim=-1)
                q_log_probs = F.log_softmax(shift_logits, dim=-1)
                kl_per_token = F.kl_div(q_log_probs, p_log_probs, reduction='none', log_target=True).sum(dim=-1)

                kl_mask = valid_mask * shift_is_general.float()
                kl_loss = (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp(min=1e-9)

            total_loss = ce_loss + self.kl_weight * kl_loss

            return (total_loss, outputs) if return_outputs else total_loss

        return outputs.loss
