import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(model: nn.Module, reference_model: nn.Module, inputs: dict, beta: float, device: torch.device) -> torch.Tensor:
    """
    Computes the Direct Preference Optimization loss for a batch of inputs with attention masks

    Args:
        model: The current policy model.
        reference_model: The reference policy model.
        inputs: A batch of inputs from the DataLoader, containing:
            - 'chosen_ids': Tensor of chosen output token IDs.
            - 'rejected_ids': Tensor of rejected output token IDs.
            - 'chosen_mask': Attention mask for chosen (1=real token, 0=pad).
            - 'rejected_mask': Attention mask for rejected (1=real token, 0=pad).
        beta: The temperature controlling strength of KL penalty.
        device: The device to perform computations on.

    Returns:
        loss: The computed DPO loss.
    """
    # Extract input tensors and move them to the specified device
    chosen_ids = inputs['chosen_ids'].to(device)
    rejected_ids = inputs['rejected_ids'].to(device)
    chosen_mask = inputs['chosen_mask'].to(device)
    rejected_mask = inputs['rejected_mask'].to(device)

    # Forward pass through models with attention masks
    chosen_outputs = model(chosen_ids, attention_mask=chosen_mask)
    rejected_outputs = model(rejected_ids, attention_mask=rejected_mask)

    # Reference model forward pass
    with torch.no_grad():
        ref_chosen_outputs = reference_model(chosen_ids, attention_mask=chosen_mask)
        ref_rejected_outputs = reference_model(rejected_ids, attention_mask=rejected_mask)

    # Helper function to compute token log probabilities with proper masking
    def compute_token_log_probs(logits, target_ids, attention_mask):
        # Shift for causal LM: predict next token
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        targets = target_ids[:, 1:]  # (B, T-1)
        mask_shifted = attention_mask[:, 1:]  # (B, T-1)
        
        # Gather log probs for target tokens
        gathered_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        
        # Mask out padded positions
        masked_log_probs = gathered_log_probs * mask_shifted.float()
        
        # Sum over sequence length (only real tokens)
        return masked_log_probs.sum(dim=1)

    # Compute log probabilities for chosen and rejected sequences
    chosen_log_prob = compute_token_log_probs(chosen_outputs.logits, chosen_ids, chosen_mask)
    rejected_log_prob = compute_token_log_probs(rejected_outputs.logits, rejected_ids, rejected_mask)
    
    ref_chosen_log_prob = compute_token_log_probs(ref_chosen_outputs.logits, chosen_ids, chosen_mask)
    ref_rejected_log_prob = compute_token_log_probs(ref_rejected_outputs.logits, rejected_ids, rejected_mask)

    # Compute DPO loss
    log_prob_diff = (chosen_log_prob - rejected_log_prob) - (ref_chosen_log_prob - ref_rejected_log_prob)
    loss = -F.logsigmoid(beta * log_prob_diff).mean()

    return loss
