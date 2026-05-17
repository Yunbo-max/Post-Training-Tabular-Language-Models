import os
import json
import argparse
import copy
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import sys
import gc
import importlib
from baselines.great.models.great import GReaT
from loss import dpo_loss

logging.basicConfig(level=logging.INFO)



# -------------------------
# Dataset (no padding here)
# -------------------------
class DPOUnconditionalDataset(Dataset):
    """
    Unconditional preference dataset: each item has chosen / rejected strings already in 'col is value' style.
    Tokenization returns variable-length tensors (no padding). Collator will pad dynamically.
    """
    def __init__(self, preference_data, tokenizer, max_length=256):
        self.data = preference_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_txt = item["chosen"]
        rejected_txt = item["rejected"]

        chosen_ids = self.tokenizer(
            chosen_txt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )["input_ids"].squeeze(0)

        rejected_ids = self.tokenizer(
            rejected_txt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )["input_ids"].squeeze(0)

        # Unconditional: empty instruction
        instruction_ids = torch.empty(0, dtype=torch.long)

        return {
            "instruction_ids": instruction_ids,  # (0,)
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids
        }

# -------------------------
# Dynamic Collator
# -------------------------
class DPODynamicCollator:
    """
    Pads variable-length chosen/rejected sequences and creates attention masks (1=real token, 0=pad).
    Instruction is empty (B x 0) for unconditional DPO.
    """
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def _pad(self, seqs):
        lens = [s.size(0) for s in seqs]
        max_len = max(lens)
        out = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :lens[i]] = s
            mask[i, :lens[i]] = 1
        return out, mask

    def __call__(self, batch):
        chosen_list = [b["chosen_ids"] for b in batch]
        rejected_list = [b["rejected_ids"] for b in batch]

        chosen_ids, chosen_mask = self._pad(chosen_list)
        rejected_ids, rejected_mask = self._pad(rejected_list)

        # Empty instruction
        instruction_ids = torch.empty((len(batch), 0), dtype=torch.long)
        instruction_mask = torch.empty((len(batch), 0), dtype=torch.long)

        return {
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask
        }

# -------------------------
# Simple Trainer
# -------------------------
class SimpleDPOTrainer:
    def __init__(self, model, ref_model, train_loader, beta=0.1, lr=5e-5, device='cuda', loss_type='dpo', retain_loader=None, 
                 npo_coeff=1.0, grad_diff_coeff=1.0, kl_coeff=1.0, 
                 kto_kl_coeff=1.0, kto_forget_coeff=1.0, kto_bias=1.0, kto_scale_factor=None):
        self.model = model
        self.ref_model = ref_model
        self.train_loader = train_loader
        self.retain_loader = retain_loader  # For DPO variants that need retain data
        self.beta = beta
        self.device = device
        self.loss_type = loss_type
        self.npo_coeff = npo_coeff
        self.grad_diff_coeff = grad_diff_coeff
        self.kl_coeff = kl_coeff
        # KTO-specific parameters
        self.kto_kl_coeff = kto_kl_coeff           # Weight for KL term (IDK/keep behavior)
        self.kto_forget_coeff = kto_forget_coeff   # Weight for forget term  
        self.kto_bias = kto_bias                   # Bias term in KTO loss
        self.kto_scale_factor = kto_scale_factor if kto_scale_factor is not None else (beta)  # Scale factor
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def get_batch_loss(self, logits, labels):
        """Helper function to compute batch loss for individual samples"""
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Compute loss for each sample
        losses = loss_fct(shift_logits, shift_labels)
        # Reshape to (batch_size, seq_len-1) and sum over sequence length
        losses = losses.view(labels.size(0), -1).sum(dim=1)
        return losses

    def compute_dpo_variants_loss(self, batch, retain_batch=None):
        """Compute DPO variants loss including gradient difference and KL regularization"""
        # Extract inputs
        chosen_ids = batch['chosen_ids'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        chosen_mask = batch['chosen_mask'].to(self.device)
        rejected_mask = batch['rejected_mask'].to(self.device)
        
        # Forward pass - current model
        chosen_outputs = self.model(chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
        rejected_outputs = self.model(rejected_ids, attention_mask=rejected_mask, labels=rejected_ids)
        
        # Forward pass - reference model
        with torch.no_grad():
            ref_chosen_outputs = self.ref_model(chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
            ref_rejected_outputs = self.ref_model(rejected_ids, attention_mask=rejected_mask, labels=rejected_ids)
            
            # Compute reference losses
            chosen_loss_ref = self.get_batch_loss(ref_chosen_outputs.logits, chosen_ids)
            rejected_loss_ref = self.get_batch_loss(ref_rejected_outputs.logits, rejected_ids)
        
        # Compute current model losses
        chosen_loss_current = self.get_batch_loss(chosen_outputs.logits, chosen_ids)
        rejected_loss_current = self.get_batch_loss(rejected_outputs.logits, rejected_ids)
        
        # DPO loss computation
        pi_logratios = -chosen_loss_current + rejected_loss_current  # Note: negative because we want log prob
        ref_logratios = -chosen_loss_ref + rejected_loss_ref
        
        dpo_loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean() * self.beta
        
        # Add regularization based on loss type
        total_loss = dpo_loss
        
        if self.loss_type == 'dpo_grad_diff' and retain_batch is not None:
            # Add gradient difference regularization
            retain_ids = retain_batch['chosen_ids'].to(self.device)  # Use chosen as retain data
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            retain_outputs = self.model(retain_ids, attention_mask=retain_mask, labels=retain_ids)
            retain_loss = retain_outputs.loss
            total_loss = total_loss + retain_loss
            
        elif self.loss_type == 'dpo_kl' and retain_batch is not None:
            # Add KL divergence regularization
            retain_ids = retain_batch['chosen_ids'].to(self.device)
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            # Reference model probabilities (target)
            with torch.no_grad():
                retain_ref_outputs = self.ref_model(retain_ids, attention_mask=retain_mask)
                retain_ref_probs = F.log_softmax(retain_ref_outputs.logits, dim=-1)
                retain_ref_probs = retain_ref_probs.view(-1, retain_ref_outputs.logits.shape[-1])
            
            # Current model probabilities
            current_retain_outputs = self.model(retain_ids, attention_mask=retain_mask)
            current_retain_probs = F.log_softmax(current_retain_outputs.logits, dim=-1)
            current_retain_probs = current_retain_probs.view(-1, current_retain_outputs.logits.shape[-1])
            
            # KL divergence loss
            kl_loss = F.kl_div(current_retain_probs, retain_ref_probs, reduction='batchmean', log_target=True)
            total_loss = total_loss + kl_loss
        
        return total_loss
    

    def compute_tabgraa_variants_loss(self, batch, retain_batch=None):
        """Compute KTO variants loss with corrected log probability calculations"""
        # For KTO, we treat chosen as "idk" (identity/keep) and rejected as "forget"
        idk_ids = batch['chosen_ids'].to(self.device)  # IDK data (keep/preserve)
        forget_ids = batch['rejected_ids'].to(self.device)  # Forget data (reduce likelihood)
        idk_mask = batch['chosen_mask'].to(self.device)
        forget_mask = batch['rejected_mask'].to(self.device)
        
        # Compute log probabilities using the helper from loss.py
        def compute_token_log_probs(logits, target_ids, attention_mask):
            """Helper to compute proper log probabilities"""
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
        
        # Current model forward pass
        idk_outputs = self.model(idk_ids, attention_mask=idk_mask)
        forget_outputs = self.model(forget_ids, attention_mask=forget_mask)
        
        # Reference model forward pass
        with torch.no_grad():
            idk_outputs_oracle = self.ref_model(idk_ids, attention_mask=idk_mask)
            forget_outputs_oracle = self.ref_model(forget_ids, attention_mask=forget_mask)
        
        # Compute log probabilities (not losses)
        idk_log_prob = compute_token_log_probs(idk_outputs.logits, idk_ids, idk_mask)
        idk_log_prob_oracle = compute_token_log_probs(idk_outputs_oracle.logits, idk_ids, idk_mask)
        
        forget_log_prob = compute_token_log_probs(forget_outputs.logits, forget_ids, forget_mask)
        forget_log_prob_oracle = compute_token_log_probs(forget_outputs_oracle.logits, forget_ids, forget_mask)
        
        # KTO formulation with tunable parameters
        KL_term = (idk_log_prob - idk_log_prob_oracle).mean()
        log_ratios = (forget_log_prob - forget_log_prob_oracle).mean()
        
        # KTO loss variants with parameterized coefficients
        if self.loss_type == 'tabgraa':
            kto_loss = self.kto_bias - F.sigmoid(self.kto_kl_coeff * KL_term - self.beta * self.kto_forget_coeff * log_ratios) * self.kto_scale_factor
        elif self.loss_type in ['tabgraa_logsigmoid', 'tabgraa_logsigmoid_grad_diff']:
            kto_loss = - F.logsigmoid(self.kto_kl_coeff * KL_term - self.beta * self.kto_forget_coeff * log_ratios) * self.kto_scale_factor
        
        total_loss = kto_loss
        
        # Add gradient difference regularization for tabgraa_logsigmoid_grad_diff
        if self.loss_type == 'tabgraa_logsigmoid_grad_diff' and retain_batch is not None:
            retain_ids = retain_batch['chosen_ids'].to(self.device)
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            retain_outputs = self.model(retain_ids, attention_mask=retain_mask, labels=retain_ids)
            retain_loss = retain_outputs.loss
            total_loss = total_loss + retain_loss
            
            print(f"KL_term: {KL_term.item():.4f}, Log_ratios: {log_ratios.item():.4f}, KTO_loss: {kto_loss.item():.4f}")
        
        return total_loss

    def compute_npo_variants_loss(self, batch, retain_batch=None):
        """Compute NPO variants loss including gradient difference and KL regularization"""
        # For NPO, we only use rejected data as "forget" data
        forget_ids = batch['rejected_ids'].to(self.device)  # Forget data (reduce likelihood)
        forget_mask = batch['rejected_mask'].to(self.device)
        
        # Forward pass - current model on forget data
        forget_outputs = self.model(forget_ids, attention_mask=forget_mask, labels=forget_ids)
        forget_loss_current = self.get_batch_loss(forget_outputs.logits, forget_ids)
        
        # Forward pass - reference model on forget data
        with torch.no_grad():
            forget_outputs_oracle = self.ref_model(forget_ids, attention_mask=forget_mask, labels=forget_ids)
            forget_loss_oracle = self.get_batch_loss(forget_outputs_oracle.logits, forget_ids)
        
        # NPO formulation: neg_log_ratios = current_loss - reference_loss
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        
        # Basic NPO loss: -logsigmoid(β * neg_log_ratios) * 2/β
        npo_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * self.beta
        
        total_loss = self.npo_coeff * npo_loss
        
        # Add regularization based on loss type
        if self.loss_type == 'npo_grad_diff' and retain_batch is not None:
            # Add gradient difference regularization
            retain_ids = retain_batch['chosen_ids'].to(self.device)
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            retain_outputs = self.model(retain_ids, attention_mask=retain_mask, labels=retain_ids)
            retain_loss = retain_outputs.loss
            total_loss = total_loss + self.grad_diff_coeff * retain_loss
            
        elif self.loss_type == 'npo_kl' and retain_batch is not None:
            # Add KL divergence regularization
            retain_ids = retain_batch['chosen_ids'].to(self.device)
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            # Reference model probabilities (target)
            with torch.no_grad():
                retain_outputs_oracle = self.ref_model(retain_ids, attention_mask=retain_mask)
                retain_probs_oracle = F.log_softmax(retain_outputs_oracle.logits, dim=-1)
                retain_probs_oracle = retain_probs_oracle.view(-1, retain_outputs_oracle.logits.shape[-1])
            
            # Current model probabilities
            current_retain_outputs = self.model(retain_ids, attention_mask=retain_mask)
            current_retain_probs = F.log_softmax(current_retain_outputs.logits, dim=-1)
            current_retain_probs = current_retain_probs.view(-1, current_retain_outputs.logits.shape[-1])
            
            # KL divergence loss (minimize KL between current and reference)
            kl_loss = F.kl_div(current_retain_probs, retain_probs_oracle, reduction='batchmean', log_target=True)
            total_loss = total_loss + self.kl_coeff * kl_loss
        
        return total_loss

    def compute_kto_variants_loss(self, batch, retain_batch=None):
        """Compute KTO variants loss including sigmoid and log-sigmoid formulations"""
        # For KTO, we treat chosen as "idk" (identity/keep) and rejected as "forget"
        idk_ids = batch['chosen_ids'].to(self.device)  # IDK data (keep/preserve)
        forget_ids = batch['rejected_ids'].to(self.device)  # Forget data (reduce likelihood)
        idk_mask = batch['chosen_mask'].to(self.device)
        forget_mask = batch['rejected_mask'].to(self.device)
        
        # Compute KL term (difference between current model and reference on IDK data)
        with torch.no_grad():
            idk_outputs = self.model(idk_ids, attention_mask=idk_mask, labels=idk_ids)
            idk_outputs_oracle = self.ref_model(idk_ids, attention_mask=idk_mask, labels=idk_ids)
            idk_loss_log = -self.get_batch_loss(idk_outputs.logits, idk_ids)
            idk_loss_log_oracle = -self.get_batch_loss(idk_outputs_oracle.logits, idk_ids)
            
            KL_term = (idk_loss_log - idk_loss_log_oracle).mean()
            
            # Reference loss on forget data
            forget_outputs_oracle = self.ref_model(forget_ids, attention_mask=forget_mask, labels=forget_ids)
            forget_loss_oracle = -self.get_batch_loss(forget_outputs_oracle.logits, forget_ids)
        
        # Current model loss on forget data
        forget_outputs = self.model(forget_ids, attention_mask=forget_mask, labels=forget_ids)
        forget_loss = -self.get_batch_loss(forget_outputs.logits, forget_ids)
        log_ratios = forget_loss - forget_loss_oracle
        
        # KTO loss variants
        if self.loss_type == 'tabgraa':
            kto_loss = 1.0 - F.sigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
        elif self.loss_type in ['tabgraa_logsigmoid', 'tabgraa_logsigmoid_grad_diff']:
            kto_loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
        else:
            kto_loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
        
        total_loss = kto_loss
        
        # Add gradient difference regularization for tabgraa_logsigmoid_grad_diff
        if self.loss_type == 'tabgraa_logsigmoid_grad_diff' and retain_batch is not None:
            retain_ids = retain_batch['chosen_ids'].to(self.device)
            retain_mask = retain_batch['chosen_mask'].to(self.device)
            
            retain_outputs = self.model(retain_ids, attention_mask=retain_mask, labels=retain_ids)
            retain_loss = retain_outputs.loss
            total_loss = total_loss + retain_loss
            
            print(f"KL_term: {KL_term.item():.4f}")  # Debug info
        
        return total_loss

    def train_one_epoch(self, epoch_num, log_interval=10):
        """Train for one epoch and return the loss"""
        self.model.train()
        total = 0.0
        
        # Create iterator for retain data if needed
        retain_iter = None
        if self.retain_loader and self.loss_type in ['dpo_grad_diff', 'dpo_kl', 'tabgraa_logsigmoid_grad_diff', 'npo_grad_diff', 'npo_kl']:
            retain_iter = iter(self.retain_loader)
        
        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get retain batch if needed
            retain_batch = None
            if retain_iter and self.loss_type in ['dpo_grad_diff', 'dpo_kl', 'tabgraa_logsigmoid_grad_diff', 'npo_grad_diff', 'npo_kl']:
                try:
                    retain_batch = next(retain_iter)
                    retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
                except StopIteration:
                    # Restart the iterator if we've gone through all retain data
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
            
            # Compute loss based on type
            if self.loss_type == 'dpo':
                loss = dpo_loss(self.model, self.ref_model, batch, self.beta, self.device)
            elif self.loss_type in ['dpo_grad_diff', 'dpo_kl']:
                loss = self.compute_dpo_variants_loss(batch, retain_batch)
            elif self.loss_type in ['tabgraa', 'tabgraa_logsigmoid', 'tabgraa_logsigmoid_grad_diff']:
                loss = self.compute_tabgraa_variants_loss(batch, retain_batch)
            elif self.loss_type in ['kto_sigmoid', 'kto_logsigmoid', 'kto_logsigmoid_grad_diff']:
                loss = self.compute_kto_variants_loss(batch, retain_batch)
            elif self.loss_type in ['npo', 'npo_grad_diff', 'npo_kl']:
                loss = self.compute_npo_variants_loss(batch, retain_batch)
            else:
                # Default fallback
                loss = dpo_loss(self.model, self.ref_model, batch, self.beta, self.device)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
            
            if (step + 1) % log_interval == 0:
                print(f"  Epoch {epoch_num}, Step {step+1}/{len(self.train_loader)}, Loss {(total/(step+1)):.4f}")
        
        epoch_loss = total / len(self.train_loader)
        print(f"  Epoch {epoch_num} completed. Avg Loss: {epoch_loss:.4f}")
        return epoch_loss

# -------------------------
# DPO Data Generation Function
# -------------------------
def generate_dpo_synthetic_data(synthetic_path, original_path, train_json_path, test_json_path, top_percent, reward_type="classifier", model_type="great", forget_cond=None):
    """Generate DPO synthetic pairs using the selected reward type.

    reward_type options:
      - 'classifier': Original RF-based distinguishability reward (generate_dpo_synthetic.py)
      - 'dcr': DCR-based reward, no classifier (generate_dpo_synthetic_dcr.py) [Option 4]
      - 'holdout': Holdout split classifier reward (generate_dpo_synthetic_holdout.py) [Option 3]
    """
    print(f"\n🔄 Generating DPO synthetic pairs...")
    print(f"  Reward type: {reward_type}")
    print(f"  Synthetic data: {synthetic_path}")
    print(f"  Original data: {original_path}")
    print(f"  Output train: {train_json_path}")
    print(f"  Top percent: {top_percent}")

    # Verify input files exist
    if not os.path.exists(synthetic_path):
        print(f"❌ Synthetic data file not found: {synthetic_path}")
        return False
    
    if not os.path.exists(original_path):
        print(f"❌ Original data file not found: {original_path}")
        return False

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(train_json_path), exist_ok=True)
    
    # Select the reward script based on reward_type
    reward_scripts = {
        'classifier': 'generate_dpo_synthetic.py',
        'classifier_leakage': 'generate_dpo_synthetic.py',  # same script but with --use_existing_synthetic
        'dcr': 'generate_dpo_synthetic_dcr.py',
        'holdout': 'generate_dpo_synthetic_holdout.py',
        'random_scoring': 'generate_dpo_synthetic_random_scoring.py',
        'random_pairing': 'generate_dpo_synthetic_random_pairing.py',
        'random_group': 'generate_dpo_synthetic_random_group.py',
        'structure': 'generate_dpo_synthetic_structure.py',
        'unlearn': 'generate_dpo_synthetic_unlearn.py',
    }
    # Generic dispatcher for C3 marginal, C5 OOD rewards, and multi-reward Pareto
    generic_rewards = {
        'marginal', 'correlation', 'density',
        'isolation_forest', 'autoencoder', 'mahalanobis', 'ocsvm', 'lof',
        'multi',
    }
    if reward_type in generic_rewards:
        script = 'generate_dpo_synthetic_reward.py'
    else:
        script = reward_scripts.get(reward_type, 'generate_dpo_synthetic.py')

    cmd = [
        sys.executable, script,
        "--synthetic", synthetic_path,
        "--original", original_path,
        "--train_json", train_json_path,
        "--test_json", test_json_path,
        "--top_percent", str(top_percent),
        "--model_type", model_type,
    ]

    # For leakage test: use existing synthetic data instead of generating new samples
    if reward_type == 'classifier_leakage':
        cmd.append("--use_existing_synthetic")

    # For the generic dispatcher, pass the reward name through
    if reward_type in generic_rewards:
        cmd.extend(["--reward", reward_type])

    # Unlearning requires a forget-condition
    if reward_type == 'unlearn':
        if not forget_cond:
            print("❌ reward_type=unlearn requires --forget_cond")
            return False
        cmd.extend(["--forget_cond", forget_cond])

    print(f"🔧 Running command: {' '.join(cmd)}")

    # Subprocess timeout: pair generation should not take longer than this even
    # for the largest dataset. If it does, the great.sample step is most likely
    # stuck in a no-progress retry loop; kill and retry with --use_existing_synthetic.
    pair_timeout_s = int(os.environ.get("PAIR_SUBPROC_TIMEOUT", "900"))

    def _try_run(cmd_):
        return subprocess.run(cmd_, capture_output=True, text=True,
                              check=True, timeout=pair_timeout_s)

    try:
        result = _try_run(cmd)
        print("✅ DPO synthetic pairs generated successfully!")
    except subprocess.TimeoutExpired as e:
        print(f"⚠️  Pair-generation subprocess exceeded {pair_timeout_s}s; retrying with --use_existing_synthetic.")
        # Add the flag if not already there (avoid duplication for classifier_leakage)
        if "--use_existing_synthetic" not in cmd:
            cmd.append("--use_existing_synthetic")
        try:
            result = _try_run(cmd)
            print("✅ DPO synthetic pairs generated (existing-synth fallback).")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
            print(f"❌ Pair generation failed even with fallback: {e2}")
            if hasattr(e2, "stdout"): print(f"STDOUT: {e2.stdout[-2000:] if e2.stdout else ''}")
            if hasattr(e2, "stderr"): print(f"STDERR: {e2.stderr[-2000:] if e2.stderr else ''}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating DPO pairs: {e}")
        print(f"STDOUT: {e.stdout[-2000:] if e.stdout else ''}")
        print(f"STDERR: {e.stderr[-2000:] if e.stderr else ''}")
        return False

    # Verify output files were created (only check train file now)
    if os.path.exists(train_json_path):
        print(f"✓ Verified: {train_json_path} exists")
        return True
    else:
        print(f"❌ Train file not created: {train_json_path}")
        return False

# -------------------------
# Synthetic Data Sampling Function
# -------------------------
def sample_synthetic_data(great_model, n_samples, synthetic_path, device):
    """Sample synthetic data and save to file"""
    print(f"\n🎲 Sampling {n_samples} synthetic samples...")
    
    try:
        samples = great_model.sample(n_samples, k=100, device=device)
        samples.to_csv(synthetic_path, index=False)
        print(f"✅ Saved synthetic samples to {synthetic_path}")
        return True
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        return False

# -------------------------
# Main Iterative Training Function
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Iterative DPO Training with Synthetic Data Regeneration")


    # Add to argument parser
    ap.add_argument('--model_type', type=str, default='great', 
                    choices=['great', 'great_gemma'], 
                    help='Model type: great (distilgpt2) or great_gemma (gemma-2b)')

    ap.add_argument('--dataname', required=True, help='Dataset name (e.g., adult)')
    ap.add_argument('--total_epochs', type=int, default=5, help='Total number of epochs to train')
    ap.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    ap.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    ap.add_argument('--beta', type=float, default=10.0, help='DPO temperature parameter')
    ap.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    ap.add_argument('--top_percent', type=float, default=100.0, help='Top percent of DPO pairs to use')
    ap.add_argument('--n_samples_per_iter', type=int, default=5, help='Number of synthetic files to generate per iteration')
    ap.add_argument('--max_samples', type=int, default=None, help='Cap on number of rows per synthetic file (default: use dataset train_num)')
    ap.add_argument('--experiment_dir', default=None, help='Experiment directory for organized storage')
    ap.add_argument('--round_number', type=int, default=1, help='Training round number for versioned storage')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output_dir', default=None, help='Output directory (default: ckpt/{dataname}/iterative_dpo_model)')
    ap.add_argument('--loss_type', type=str, default='dpo', 
                    choices=['dpo', 'dpo_grad_diff', 'dpo_kl', 'tabgraa', 'tabgraa_logsigmoid', 'tabgraa_logsigmoid_grad_diff', 
                             'npo', 'npo_grad_diff', 'npo_kl','kto_sigmoid', 'kto_logsigmoid', 'kto_logsigmoid_grad_diff'], 
                    help='Loss type: dpo (standard), dpo_grad_diff (DPO + gradient difference), dpo_kl (DPO + KL divergence), ' +
                         'tabgraa (KTO with sigmoid), tabgraa_logsigmoid (KTO with log sigmoid), ' +
                         'tabgraa_logsigmoid_grad_diff (KTO with log sigmoid + gradient difference), ' +
                         'npo (Negative Preference Optimization), npo_grad_diff (NPO + gradient difference), ' +
                         'npo_kl (NPO + KL divergence)')
    ap.add_argument('--reward_type', type=str, default='classifier',
                    choices=['classifier', 'classifier_leakage', 'dcr', 'holdout',
                             'random_scoring', 'random_pairing', 'random_group',
                             'structure', 'unlearn',
                             'marginal', 'correlation', 'density',
                             'isolation_forest', 'autoencoder', 'mahalanobis', 'ocsvm', 'lof',
                             'multi'],
                    help='Reward signal for DPO pair generation: '
                         'classifier (RF distinguishability, default), '
                         'dcr (DCR distance, no classifier - Option 4), '
                         'random_scoring (random rewards - ablation), '
                         'random_pairing (RF scoring + random pairs - ablation), '
                         'holdout (holdout split classifier - Option 3)')
    ap.add_argument('--no_shuffle', action='store_true', default=False,
                    help='Disable batch shuffling (ordered by contrast level) for ablation')
    ap.add_argument('--npo_coeff', type=float, default=1.0, help='NPO loss coefficient for NPO variants')
    ap.add_argument('--grad_diff_coeff', type=float, default=1.0, help='Gradient difference coefficient for NPO+grad_diff')
    ap.add_argument('--kl_coeff', type=float, default=1.0, help='KL divergence coefficient for NPO+KL')
    # KTO-specific parameters
    ap.add_argument('--kto_kl_coeff', type=float, default=1.0, help='KTO: Weight for KL term (IDK/keep behavior)')
    ap.add_argument('--kto_forget_coeff', type=float, default=1.0, help='KTO: Weight for forget term (reduce likelihood)')
    ap.add_argument('--kto_bias', type=float, default=2.0, help='KTO: Bias term in loss function')
    ap.add_argument('--kto_scale_factor', type=float, default=None, help='KTO: Scale factor (default: 2/beta)')
    ap.add_argument('--forget_cond', type=str, default=None,
                    help="For reward_type=unlearn: pandas-eval string selecting the forget subgroup, "
                         "e.g. \"education == 'Doctorate'\"")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Using device:", device)
    print(f"🚀 Starting iterative DPO training for {args.total_epochs} epochs (Round {args.round_number})")
    print(f"📊 Loss type: {args.loss_type}")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up experiment directory structure
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        # Create subdirectories
        os.makedirs(f"{experiment_dir}/synthetic_data", exist_ok=True)
        os.makedirs(f"{experiment_dir}/training_logs", exist_ok=True)
        os.makedirs(f"{experiment_dir}/models", exist_ok=True)
        
        # Create iteration subdirectories
        for epoch in range(1, args.total_epochs + 1):
            os.makedirs(f"{experiment_dir}/synthetic_data/iteration_{epoch}", exist_ok=True)
        
        
        print(f"📁 Experiment directory: {experiment_dir}")
    else:
        experiment_dir = None
    
    # Set up paths
    if args.output_dir is None:
        if experiment_dir:
            args.output_dir = f'{experiment_dir}/models/final_dpo_model'
        else:
            args.output_dir = f'ckpt/{args.dataname}/iterative_dpo_model'


    # Load the appropriate model based on model_type
    if args.model_type == 'great':
        from baselines.great.models.great import GReaT
        model_class = GReaT
        base_model_name = "distilgpt2"
    elif args.model_type == 'great_gemma':
        from baselines.great_gemma.models.great import GReaT
        model_class = GReaT
        # base_model_name = "google/gemma-2b"
        # base_model_name = "gpt2"
        # base_model_name = 'EleutherAI/gpt-neo-125M'
        base_model_name = 'gpt2-large'
        
    
    # Set model-specific paths
    base_ckpt_dir = os.path.join(curr_dir, "baselines", args.model_type, "ckpt", args.dataname)
    base_model_path = os.path.join(base_ckpt_dir, "model.pt")
    original_data_path = os.path.join(curr_dir, "synthetic", args.dataname, "real.csv")
    synthetic_dir = os.path.join(curr_dir, "synthetic", args.dataname)


    
    print(f"📍 Using round {args.round_number} configuration:")
    print(f"  - Base model: {base_model_path}")
    print(f"  - Original data: {original_data_path}")
    print(f"  - Synthetic dir: {synthetic_dir}")
    
    # Load GReaT base model
    print("\n📂 Loading base GReaT model...")
    # great = GReaT("distilgpt2")
    great = model_class(base_model_name)
    if os.path.exists(base_model_path):
        great.load_finetuned_model(base_model_path)
        print(f"✅ Loaded base fine-tuned model: {base_model_path}")
    else:
        print("⚠️ Base model.pt not found. Using raw distilgpt2.")

    model = great.model.to(device)
    tokenizer = great.tokenizer

    # Build the FIXED reference model once (paper Algorithm 1: π_ref ← π_θ_0, fixed throughout).
    # This is a frozen deepcopy of the SFT'd base model. We re-use it across all iterations
    # rather than re-creating from `model` (which evolves) on every iteration.
    print("📌 Creating fixed reference model (π_ref) from SFT'd base — kept frozen across iterations.")
    fixed_ref_model = copy.deepcopy(model).to(device)
    fixed_ref_model.eval()
    for _p in fixed_ref_model.parameters():
        _p.requires_grad = False

    # Restore column info using training data
    dataset_path = os.path.join(curr_dir, "data", args.dataname, "train.csv")
    if os.path.exists(dataset_path):
        train_df = pd.read_csv(dataset_path)
        from baselines.great.models.great_utils import _array_to_dataframe
        df = _array_to_dataframe(train_df, columns=None)
        great._update_column_information(df)
        great._update_conditional_information(df, conditional_col=None)
        print("✅ Updated column information from training data")
    else:
        print("⚠️ Warning: train.csv not found; sampling may fail.")

    # Load dataset info to get the correct number of samples
    info_path = os.path.join(curr_dir, "data", args.dataname, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        n_samples = info['train_num']
        if args.max_samples and args.max_samples < n_samples:
            print(f"✅ Dataset has {n_samples} rows, capping at {args.max_samples} per file")
            n_samples = args.max_samples
        else:
            print(f"✅ Will generate {n_samples} samples per iteration")
    else:
        n_samples = 1000  # fallback
        print(f"⚠️ info.json not found, using fallback of {n_samples} samples")

    # Initialize tracking variables
    all_losses = []
    
    # Initial synthetic data path (use existing synthetic data if available)
    # Default synthetic filename based on model type
    if args.model_type == 'great':
        default_synthetic_name = "great.csv"
    else:
        default_synthetic_name = "great_gemma.csv"
    
    # Update synthetic path usage
    current_synthetic_path = os.path.join(synthetic_dir, default_synthetic_name)
    

    # current_synthetic_path = os.path.join(synthetic_dir, "great.csv")
    if not os.path.exists(current_synthetic_path):
        # Generate initial synthetic data
        print("\n🎯 No existing synthetic data found. Generating initial synthetic data...")
        if not sample_synthetic_data(great, n_samples, current_synthetic_path, device):
            print("❌ Failed to generate initial synthetic data. Exiting.")
            return
    else:
        print(f"✅ Using existing synthetic data: {current_synthetic_path}")

    # Iterative training loop
    for epoch in range(1, args.total_epochs + 1):
        print(f"\n" + "="*60)
        print(f"🔄 ITERATION {epoch}/{args.total_epochs}")
        print(f"="*60)
        
        # Step 1: Generate DPO pairs using current synthetic data
        # Save pairs in the experiment directory with iteration numbering
        pairs_dir = os.path.join(args.experiment_dir, "pairs_data")
        os.makedirs(pairs_dir, exist_ok=True)
        
        train_json_path = os.path.join(pairs_dir, f"train_pairs_iter_{epoch}.json")
        test_json_path = os.path.join(pairs_dir, f"test_pairs_iter_{epoch}.json")
        
        success = generate_dpo_synthetic_data(
            synthetic_path=current_synthetic_path,
            original_path=original_data_path,
            train_json_path=train_json_path,
            test_json_path=test_json_path,
            top_percent=args.top_percent,
            reward_type=args.reward_type,
            model_type=args.model_type,
            forget_cond=getattr(args, 'forget_cond', None),
        )
        
        if not success:
            print(f"❌ Failed to generate DPO pairs for epoch {epoch}. Skipping...")
            continue
        
        # Log pairs data locations
        print(f"📁 DPO pairs saved:")
        print(f"  Train pairs: {train_json_path}")
        print(f"  Note: Using ALL pairs for training (no test split)")
        
        # Step 2: Load preference data and create datasets
        print(f"\n📚 Loading preference data for epoch {epoch}...")
        try:
            with open(train_json_path, 'r') as f:
                train_pref = json.load(f)
            print(f"✅ Loaded {len(train_pref)} training preference pairs")
        except Exception as e:
            print(f"❌ Failed to load preference data: {e}")
            continue
        
        # Build datasets
        train_dataset = DPOUnconditionalDataset(train_pref, tokenizer, max_length=args.max_length)
        
        # Create retain dataset for DPO variants (using original data)
        retain_loader = None
        if args.loss_type in ['dpo_grad_diff', 'dpo_kl', 'tabgraa_logsigmoid_grad_diff', 'npo_grad_diff', 'npo_kl']:
            # Create retain dataset from original real data
            retain_pref = []
            # Convert original data to preference format for retain (chosen = rejected = original)
            try:
                # Read original data and convert to "retention" pairs
                original_df = pd.read_csv(original_data_path)
                for _, row in original_df.head(min(len(train_pref), 100)).iterrows():  # Limit retain data size
                    # Convert row to text format (similar to synthetic data)
                    row_text = ", ".join([f"{col} is {val}" for col, val in zip(original_df.columns, row)])
                    retain_pref.append({
                        "chosen": row_text,
                        "rejected": row_text  # Same for retention - we want to preserve this distribution
                    })
                
                if retain_pref:
                    retain_dataset = DPOUnconditionalDataset(retain_pref, tokenizer, max_length=args.max_length)
                    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
                    print(f"✅ Created retain dataset with {len(retain_pref)} samples for {args.loss_type}")
                else:
                    print(f"⚠️ No retain data created for {args.loss_type}")
            except Exception as e:
                print(f"⚠️ Failed to create retain dataset: {e}. Continuing without retain data.")
        
        # Collator & loader
        collator = DPODynamicCollator(pad_id=tokenizer.pad_token_id)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle, collate_fn=collator)
        
        # Step 3: Use the FIXED reference model created once at startup.
        # (Per paper Algorithm 1, π_ref stays frozen at the SFT state across all iterations.
        # Previously this was deepcopying the current policy each iter, which caused unlearning
        # loss to drift because the reference kept chasing the policy.)
        ref_model = fixed_ref_model
        
        # Step 4: Train for one epoch
        print(f"\n🏋️ Training DPO for epoch {epoch}...")
        
        # Monitor GPU memory before training
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated(device) / 1024**3  # GB
            print(f"📊 GPU memory before training: {gpu_memory_before:.2f} GB")
        
        trainer = SimpleDPOTrainer(
            model=model,
            ref_model=ref_model,
            train_loader=train_loader,
            retain_loader=retain_loader,
            beta=args.beta,
            lr=args.learning_rate,
            device=device,
            loss_type=args.loss_type,
            npo_coeff=args.npo_coeff,
            grad_diff_coeff=args.grad_diff_coeff,
            kl_coeff=args.kl_coeff,
            kto_kl_coeff=args.kto_kl_coeff,
            kto_forget_coeff=args.kto_forget_coeff,
            kto_bias=args.kto_bias,
            kto_scale_factor=args.kto_scale_factor
        )
        
        epoch_loss = trainer.train_one_epoch(epoch)
        all_losses.append(epoch_loss)
        
        # Update the GReaT model with trained weights
        great.model = model
        
        # Step 5: Generate multiple synthetic data files for current iteration
        print(f"\n🎲 Generating {args.n_samples_per_iter} synthetic data files for iteration {epoch}...")
        
        iteration_synthetic_files = []
        for sample_idx in range(1, args.n_samples_per_iter + 1):
            if experiment_dir:
                # Save in experiment directory structure
                iter_synthetic_path = f"{experiment_dir}/synthetic_data/iteration_{epoch}/dpo_syn_iter_{epoch}_{sample_idx}.csv"
            else:
                # Fallback to original location
                iter_synthetic_path = os.path.join(synthetic_dir, f"dpo_syn_iter_{epoch}_{sample_idx}.csv")
            
            success = sample_synthetic_data(great, n_samples, iter_synthetic_path, device)
            if success:
                iteration_synthetic_files.append(iter_synthetic_path)
                print(f"✅ Generated sample {sample_idx}/{args.n_samples_per_iter}: {iter_synthetic_path}")
            else:
                print(f"❌ Failed to generate sample {sample_idx}/{args.n_samples_per_iter}")
        
        # Update current synthetic path for next iteration (use first generated file)
        if epoch < args.total_epochs and iteration_synthetic_files:
            current_synthetic_path = iteration_synthetic_files[0]
            print(f"✅ Using {current_synthetic_path} for next iteration")
        elif epoch < args.total_epochs:
            print(f"⚠️ No new synthetic data generated. Using previous: {current_synthetic_path}")
        
        # Clean up preference files to save space - DISABLED to preserve pairs for analysis
        # try:
        #     os.remove(train_json_path)
        #     os.remove(test_json_path)
        #     print(f"🧹 Cleaned up preference files for epoch {epoch}")
        # except:
        #     pass
        print(f"📁 Keeping DPO pairs file for analysis:")
        print(f"  Train pairs: {train_json_path}")
        print(f"  Note: Contains ALL pairs (no test split needed)")
        
        # Step 6: GPU Memory Cleanup
        print(f"🧹 Cleaning up GPU memory after epoch {epoch}...")
        
        # Delete reference model to free memory
        del ref_model
        del trainer
        
        # Clear DataLoader to release memory
        del train_loader
        del train_dataset
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Monitor GPU memory after cleanup
            gpu_memory_after = torch.cuda.memory_allocated(device) / 1024**3  # GB
            print(f"📊 GPU memory after cleanup: {gpu_memory_after:.2f} GB")
            
        print(f"✅ GPU memory cleanup completed for epoch {epoch}")
    
    # Step 6: Save final model
    print(f"\n💾 Saving final iterative DPO model...")
    os.makedirs(args.output_dir, exist_ok=True)
    great.save(args.output_dir)
    print(f"✅ Final model saved to {args.output_dir}")
    
    # Step 7: Generate final synthetic samples (use last iteration number)
    print(f"\n🎯 Generating {args.n_samples_per_iter} final synthetic samples (iteration_{args.total_epochs})...")
    final_synthetic_files = []
    
    # Use the last iteration's files as the "final" files
    if experiment_dir and args.total_epochs > 0:
        # The final samples are already generated in the last iteration
        final_iteration_dir = f"{experiment_dir}/synthetic_data/iteration_{args.total_epochs}"
        if os.path.exists(final_iteration_dir):
            final_files = [f for f in os.listdir(final_iteration_dir) if f.endswith('.csv')]
            for f in final_files:
                final_synthetic_files.append(os.path.join(final_iteration_dir, f))
            print(f"✅ Final synthetic samples are in iteration_{args.total_epochs}: {len(final_files)} files")
        else:
            print(f"⚠️ Final iteration directory not found: {final_iteration_dir}")
    else:
        # Fallback: generate final samples in old format for backward compatibility
        for i in range(1, args.n_samples_per_iter + 1):
            final_synthetic_path = os.path.join(synthetic_dir, f"iterative_dpo_syn{i}.csv")
            success = sample_synthetic_data(great, n_samples, final_synthetic_path, device)
            if success:
                final_synthetic_files.append(final_synthetic_path)
                print(f"✅ Generated final sample {i}/{args.n_samples_per_iter}: {final_synthetic_path}")
    
    # Save experiment summary
    if experiment_dir:
        # Collect pairs files information
        pairs_dir = os.path.join(experiment_dir, "pairs_data")
        pairs_files = []
        if os.path.exists(pairs_dir):
            pairs_files = [f for f in os.listdir(pairs_dir) if f.endswith('.json')]
            pairs_files = sorted(pairs_files)
        
        summary = {
            'dataset': args.dataname,
            'round_number': args.round_number,
            'total_epochs': args.total_epochs,
            'learning_rate': args.learning_rate,
            'beta': args.beta,
            'top_percent': args.top_percent,
            'n_samples_per_iter': args.n_samples_per_iter,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'training_losses': all_losses,
            'final_synthetic_files': final_synthetic_files,
            'total_iterations': args.total_epochs,
            'samples_per_iteration': n_samples,
            'base_model_path': base_model_path,
            'synthetic_directory': synthetic_dir,
            'pairs_files': pairs_files,
            'pairs_directory': pairs_dir
        }
        
        summary_path = f"{experiment_dir}/experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Experiment summary saved: {summary_path}")
    
    # Step 8: Plot training loss
    if all_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(all_losses) + 1), all_losses, 'b-o', linewidth=2, markersize=8)
        plt.xlabel("Iteration (Epoch)", fontsize=12)
        plt.ylabel("DPO Loss", fontsize=12)
        plt.title(f"Iterative DPO Training Loss - {args.dataname.upper()}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plots in both model dir and experiment dir
        loss_plot_path = os.path.join(args.output_dir, "iterative_loss_plot.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved loss plot to {loss_plot_path}")
        
        if experiment_dir:
            exp_loss_plot_path = f"{experiment_dir}/training_logs/loss_plot.png"
            plt.savefig(exp_loss_plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved loss plot to experiment dir: {exp_loss_plot_path}")
        
        # Save loss data
        loss_data = {
            'epochs': list(range(1, len(all_losses) + 1)),
            'losses': all_losses,
            'parameters': {
                'learning_rate': args.learning_rate,
                'beta': args.beta,
                'top_percent': args.top_percent,
                'batch_size': args.batch_size
            }
        }
        loss_json_path = os.path.join(args.output_dir, "training_losses.json")
        with open(loss_json_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        print(f"✅ Saved loss data to {loss_json_path}")
        
        if experiment_dir:
            exp_loss_json_path = f"{experiment_dir}/training_logs/training_losses.json"
            with open(exp_loss_json_path, 'w') as f:
                json.dump(loss_data, f, indent=2)
            print(f"✅ Saved loss data to experiment dir: {exp_loss_json_path}")
    
    print(f"\n🎉 Iterative DPO training completed!")
    print(f"📊 Total epochs: {args.total_epochs}")
    print(f"📈 Final loss: {all_losses[-1]:.4f}")
    print(f"📁 Model saved: {args.output_dir}")
    
    if experiment_dir:
        print(f"\n📋 Experiment Results:")
        print(f"  📁 Experiment directory: {experiment_dir}")
        print(f"  🎯 Final synthetic files: {experiment_dir}/synthetic_data/iteration_{args.total_epochs}/")
        print(f"  🔄 All iteration files: {experiment_dir}/synthetic_data/iteration_*/")
        print(f"  📊 Training logs: {experiment_dir}/training_logs/")
        print(f"  🤖 Models: {experiment_dir}/models/")
        print(f"  � DPO pairs data: {experiment_dir}/pairs_data/")
        print(f"  �📋 Summary: {experiment_dir}/experiment_summary.json")
        
        # List all pairs files that were created
        pairs_dir = os.path.join(experiment_dir, "pairs_data")
        if os.path.exists(pairs_dir):
            pairs_files = [f for f in os.listdir(pairs_dir) if f.endswith('.json')]
            if pairs_files:
                print(f"\n📝 DPO Pairs Files Created:")
                for pairs_file in sorted(pairs_files):
                    pairs_path = os.path.join(pairs_dir, pairs_file)
                    file_size = os.path.getsize(pairs_path)
                    print(f"    - {pairs_file} ({file_size:,} bytes)")
        
        # Print directory structure
        print(f"\n📂 Generated Directory Structure:")
        for epoch in range(1, args.total_epochs + 1):
            iter_dir = f"{experiment_dir}/synthetic_data/iteration_{epoch}/"
            if os.path.exists(iter_dir):
                files = [f for f in os.listdir(iter_dir) if f.endswith('.csv')]
                status = " (FINAL)" if epoch == args.total_epochs else ""
                print(f"  📁 iteration_{epoch}/: {len(files)} files{status}")
    else:
        print(f"📄 Synthetic files: synthetic/{args.dataname}/iterative_dpo_syn*.csv")

if __name__ == "__main__":
    main()