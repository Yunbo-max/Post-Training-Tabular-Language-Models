# Post-Training for Tabular Language Models

Reference implementation of an iterative **reward-guided post-training**
framework for tabular language-model generators (**TabGRAA** — Tabular
Group-Relative Advantage Alignment).

Starting from a supervised fine-tuned tabular language model (default
backbone: GReaT / DistilGPT-2), the loop repeatedly samples synthetic rows,
scores them with a task-specified reward, partitions them into high- and
low-reward groups, and updates the generator with a group-relative
policy/reference objective. The same iterative pipeline also runs adapted
**DPO**, **NPO**, and **KTO** baselines and supports KL / gradient-difference
retain-set regularizers.


## Highlights

- **Iterative reward-guided alignment** — sample → score → group → update,
  repeated for $T$ rounds with a fixed SFT reference.
- **Group-relative advantage objective** — mean over reward strata *before*
  the log-sigmoid, an inductive bias for distributional alignment.
- **Reward programmability** — swap in classifier, DCR, held-out, structural,
  or constraint-style (forget-region) rewards without retraining.
- **Deployment-style studies** — target-distribution adaptation and
  forget-region suppression.
- **Full evaluation suite** — fidelity (CDE/PCC/α/β), utility (MLE),
  detection (C2ST, DA), and privacy-oriented indistinguishability (MIA).


## Repository layout

```
.
├── iterative_dpo_trainer.py            # Main iterative post-training loop
├── loss.py                             # Loss helpers
├── dataset.py                          # Dataset wrappers
├── reward_scorers.py                   # Pluggable reward scorers
├── utils.py                            # Shared utilities
│
├── generate_dpo_synthetic.py           # Default classifier-reward generation
├── generate_dpo_synthetic_dcr.py       # DCR (classifier-free) reward
├── generate_dpo_synthetic_holdout.py   # Held-out classifier reward
├── generate_dpo_synthetic_random_*.py  # Random-scoring / random-group ablations
├── generate_dpo_synthetic_structure.py # Structural-fidelity reward
├── generate_dpo_synthetic_unlearn.py   # Forget-region suppression reward
│
├── baselines/
│   └── great/                          # GReaT backbone (DistilGPT-2 tabular LM)
│       ├── main.py / sample.py / utils.py / post_process.py
│       └── models/
│
├── eval/                               # Full evaluation suite
│   ├── eval_quality.py                 # CDE / PCC / α / β
│   ├── eval_density.py                 # Marginal / joint density
│   ├── eval_detection.py               # C2ST / DA AUC
│   ├── eval_mle.py                     # Downstream ML utility
│   ├── eval_dcr.py                     # Distance to Closest Record
│   ├── eval_privacy.py                 # Aggregate privacy metrics
│   ├── eval_mia_proper.py              # MIA AUC
│   ├── eval_holdout_classifier.py
│   ├── eval_statistical_distances.py
│   ├── eval_structural_fidelity.py
│   └── {density,mle,quality,statistical}/   # per-dataset configs
│
├── privacy/                            # Standalone privacy attack scripts
│   ├── privacy_attack.py
│   └── privacy_mia_standalone.py
│
├── data_tools/                         # Dataset download / process / clean
│   ├── download_dataset.py
│   ├── process_dataset.py
│   ├── clean_dataset.py
│   └── alpaca_template.py
│
├── scripts/                            # Reproducibility shell scripts
│   ├── run_full_pipeline.sh
│   ├── run_loss_variants.sh
│   ├── run_reward_dcr.sh
│   └── run_reward_holdout.sh
│
├── data/Info/                          # Dataset schema definitions
│
├── docs/                               # Extended documentation
├── README.md  ·  LICENSE  ·  requirements.txt  ·  .gitignore
```


## Installation

```bash
git clone <this-repo>
cd Post-Training-Tabular-Language-Models
pip install -r requirements.txt
```

Tested with Python 3.9–3.11, PyTorch ≥ 2.0, Transformers ≥ 4.30.


## Datasets

The default benchmarks are five mixed-type tabular datasets: **Adult**,
**Default**, **Magic**, **Shoppers**, and **Beijing**. Schema definitions are
provided in `data/Info/*.json`.

```bash
# Download raw CSV
python data_tools/download_dataset.py --dataset adult

# Pre-process into the expected layout
python data_tools/process_dataset.py  --dataset adult
```


## Quick start

### 1. Supervised fine-tune the GReaT backbone

```bash
python baselines/great/main.py \
    --dataname adult \
    --epochs 100 \
    --batch_size 16
```

### 2. Iterative TabGRAA post-training

```bash
python iterative_dpo_trainer.py \
    --dataname adult \
    --loss_type tabgraa \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 5e-7 \
    --beta 1.0 \
    --top_percent 100 \
    --n_samples_per_iter 10
```

### 3. Evaluate

```bash
python eval/eval_quality.py            --dataname adult --syn_path <synthetic.csv>
python eval/eval_density.py            --dataname adult --syn_path <synthetic.csv>
python eval/eval_detection.py          --dataname adult --syn_path <synthetic.csv>
python eval/eval_mle.py                --dataname adult --syn_path <synthetic.csv>
python eval/eval_statistical_distances.py --dataname adult --syn_path <synthetic.csv>
python eval/eval_mia_proper.py         --dataname adult --syn_path <synthetic.csv>
```


## Supported objectives (`--loss_type`)

| `--loss_type`                       | Method                                                  |
|-------------------------------------|---------------------------------------------------------|
| `tabgraa`                           | **TabGRAA base** (sigmoid form, default)                |
| `tabgraa_logsigmoid`                | TabGRAA log-sigmoid form                                |
| `tabgraa_logsigmoid_grad_diff`      | TabGRAA log-sigmoid + retain-anchor gradient diff       |
| `dpo`                               | TabDPO baseline                                         |
| `dpo_kl`                            | TabDPO + retain-anchor KL                               |
| `dpo_grad_diff`                     | TabDPO + retain-anchor gradient diff                    |
| `npo`                               | TabNPO baseline                                         |
| `npo_kl`                            | TabNPO + retain-anchor KL                               |
| `npo_grad_diff`                     | TabNPO + retain-anchor gradient diff                    |
| `kto`                               | TabKTO baseline                                         |

For variants with `_kl` / `_grad_diff` / `tabgraa_logsigmoid_grad_diff`, the
trainer auto-builds a retain (anchor) batch from the first 100 rows of the
real training CSV — see the appendix in the paper for details.


## Reward-substitution variants

```bash
# DCR-based reward (classifier-free)
python generate_dpo_synthetic_dcr.py     --dataname adult --loss_type tabgraa

# Held-out classifier reward (scorer never sees the rows it evaluates)
python generate_dpo_synthetic_holdout.py --dataname adult --loss_type tabgraa

# Structural-fidelity reward (joint-distribution preservation)
python generate_dpo_synthetic_structure.py --dataname adult --loss_type tabgraa

# Forget-region suppression
python generate_dpo_synthetic_unlearn.py --dataname adult --loss_type tabgraa \
    --forget_cond "age >= 65"
```

Random-control ablations:

```bash
python generate_dpo_synthetic_random_scoring.py  --dataname adult --loss_type tabgraa
python generate_dpo_synthetic_random_group.py    --dataname adult --loss_type tabgraa
```


## Reproducing the paper results

End-to-end sweeps with the provided shell scripts:

```bash
# Table: loss-variant comparison, averaged across 5 benchmarks
for ds in adult default magic shoppers beijing; do
    bash scripts/run_loss_variants.sh --dataname "$ds"
done

# Reward-substitution: DCR / held-out
bash scripts/run_reward_dcr.sh     --dataname adult
bash scripts/run_reward_holdout.sh --dataname adult
```

Synthetic outputs are written under `synthetic/<dataset>/` and consumed by
the evaluation scripts above.


## Privacy attacks (optional)

```bash
python privacy/privacy_attack.py        --dataname adult --syn_path <synthetic.csv>
python privacy/privacy_mia_standalone.py --dataname adult --syn_path <synthetic.csv>
```


## Citation

```bibtex
@article{tabgraa2026,
  title  = {Self-Improving Tabular Language Models via
            Iterative Reward-Guided Alignment},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```


## License

Released under the MIT License — see [LICENSE](LICENSE).
