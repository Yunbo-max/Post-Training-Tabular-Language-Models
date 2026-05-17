#!/bin/bash

# Run TabDPO Pipeline with Holdout Split Reward (Option 3)
# Splits real data 50/50: trains reward classifier on split_A,
# evaluates with independent classifier on split_B.
#
# Usage: ./run_reward_holdout.sh --folder_name <name> [OPTIONS]

set -e

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Default parameters
DATANAME_LIST=("adult")
MODEL_TYPE="great"
TOTAL_EPOCHS=1
LEARNING_RATE_LIST=("1e-7")
BETA_LIST=(1)
TOP_PERCENT_LIST=(100)
BATCH_SIZE_LIST=(4)  # Keep small for 4090 (24GB). Use 2 for great_gemma.
N_SAMPLES_PER_ITER=1
RUN_EVALUATION=true
RETRAIN=false
RESAMPLE=false
CUSTOM_FOLDER_NAME="holdout_reward_experiment"
REWARD_TYPE="holdout"

# Loss types to test
LOSS_TYPES=(
    "tabgraa"
    # "tabgraa_logsigmoid"
    # "dpo"
    # "npo"
)

# GPU device selection
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    GPU_ID="$CUDA_VISIBLE_DEVICES"
else
    GPU_ID="0"
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --folder_name) CUSTOM_FOLDER_NAME="$2"; shift 2 ;;
        --model|--model_type) MODEL_TYPE="$2"; shift 2 ;;
        --dataname) IFS=',' read -ra DATANAME_LIST <<< "$2"; shift 2 ;;
        --datanames) IFS=',' read -ra DATANAME_LIST <<< "$2"; shift 2 ;;
        --epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --lr|--learning_rate) IFS=',' read -ra LEARNING_RATE_LIST <<< "$2"; shift 2 ;;
        --lr_list) IFS=',' read -ra LEARNING_RATE_LIST <<< "$2"; shift 2 ;;
        --beta) IFS=',' read -ra BETA_LIST <<< "$2"; shift 2 ;;
        --beta_list) IFS=',' read -ra BETA_LIST <<< "$2"; shift 2 ;;
        --top_percent) IFS=',' read -ra TOP_PERCENT_LIST <<< "$2"; shift 2 ;;
        --top_percent_list) IFS=',' read -ra TOP_PERCENT_LIST <<< "$2"; shift 2 ;;
        --batch_size) IFS=',' read -ra BATCH_SIZE_LIST <<< "$2"; shift 2 ;;
        --batch_size_list) IFS=',' read -ra BATCH_SIZE_LIST <<< "$2"; shift 2 ;;
        --n_samples_per_iter) N_SAMPLES_PER_ITER="$2"; shift 2 ;;
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --loss_types) IFS=',' read -ra LOSS_TYPES <<< "$2"; shift 2 ;;
        --no-eval) RUN_EVALUATION=false; shift ;;
        --retrain) RETRAIN=true; shift ;;
        --resample) RESAMPLE=true; shift ;;
        --help|-h)
            echo "Run TabDPO Pipeline with Holdout Split Reward (Option 3)"
            echo ""
            echo "Addresses evaluation circularity: reward classifier trained on split_A,"
            echo "evaluation classifier trained independently on split_B."
            echo ""
            echo "Usage: $0 --folder_name <name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --folder_name <name>   Custom name for experiment folder"
            echo "  --model <type>         Model type: great or great_gemma (default: great)"
            echo "  --dataname <name>      Dataset or comma-separated list (default: adult)"
            echo "  --epochs <num>         DPO training epochs (default: 1)"
            echo "  --lr <rate>            Learning rate(s) (default: 1e-7)"
            echo "  --beta <value>         Beta parameter(s) (default: 1)"
            echo "  --top_percent <num>    Top percent of pairs (default: 100)"
            echo "  --batch_size <size>    Batch size(s) (default: 6)"
            echo "  --loss_types <list>    Loss types (default: tabgraa)"
            echo "  --no-eval              Skip evaluation"
            echo "  --retrain              Force retrain base model"
            echo "  --resample             Force resample synthetic data"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [[ "$MODEL_TYPE" != "great" ]] && [[ "$MODEL_TYPE" != "great_gemma" ]]; then
    echo "Invalid model type: $MODEL_TYPE"; exit 1
fi

echo "============================================================="
echo "  HOLDOUT SPLIT REWARD PIPELINE (Option 3)"
echo "============================================================="
echo "  Reward type: Holdout (RF on split_A, eval on split_B)"
echo "  Model: $MODEL_TYPE"
echo "  Dataset(s): ${DATANAME_LIST[*]}"
echo "  Folder: $CUSTOM_FOLDER_NAME"
echo "  Epochs: $TOTAL_EPOCHS"
echo "  LR(s): ${LEARNING_RATE_LIST[*]}"
echo "  Beta(s): ${BETA_LIST[*]}"
echo "  Loss types: ${LOSS_TYPES[*]}"
echo "============================================================="

TOTAL_LR=${#LEARNING_RATE_LIST[@]}
TOTAL_BETA=${#BETA_LIST[@]}
TOTAL_TOP_PERCENT=${#TOP_PERCENT_LIST[@]}
TOTAL_BATCH_SIZE=${#BATCH_SIZE_LIST[@]}
TOTAL_LOSS_TYPES=${#LOSS_TYPES[@]}
TOTAL_COMBINATIONS=$((TOTAL_LR * TOTAL_BETA * TOTAL_TOP_PERCENT * TOTAL_BATCH_SIZE * TOTAL_LOSS_TYPES))

for DATANAME in "${DATANAME_LIST[@]}"; do

    echo ""
    echo "PROCESSING DATASET: $DATANAME"

    BASE_MODEL_DIR="baselines/$MODEL_TYPE/ckpt/$DATANAME"
    BASE_MODEL_PATH="$BASE_MODEL_DIR/model.pt"
    SYNTHETIC_DIR="synthetic/$DATANAME"
    ORIGINAL_SYNTHETIC_PATH="$SYNTHETIC_DIR/$MODEL_TYPE.csv"

    # Step 1: Train base model if needed
    if [ "$RETRAIN" = true ] || [ ! -f "$BASE_MODEL_PATH" ]; then
        echo "Training base $MODEL_TYPE model..."
        mkdir -p "$BASE_MODEL_DIR"
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode train --bs 16
        [ -f "$BASE_MODEL_PATH" ] || { echo "Training failed"; exit 1; }
    else
        echo "Base model found: $BASE_MODEL_PATH"
    fi

    # Step 2: Generate initial synthetic data if needed
    if [ "$RESAMPLE" = true ] || [ ! -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
        echo "Generating initial synthetic data..."
        mkdir -p "$SYNTHETIC_DIR"
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode sample
        [ -f "$ORIGINAL_SYNTHETIC_PATH" ] || { echo "Sampling failed"; exit 1; }
    else
        echo "Synthetic data found: $ORIGINAL_SYNTHETIC_PATH"
    fi

    MAIN_OUTPUT_DIR="synthetic/$DATANAME/$CUSTOM_FOLDER_NAME"
    mkdir -p "$MAIN_OUTPUT_DIR"

    EXPERIMENT_COUNT=0

    for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
      for LEARNING_RATE in "${LEARNING_RATE_LIST[@]}"; do
        for BETA in "${BETA_LIST[@]}"; do
          for TOP_PERCENT in "${TOP_PERCENT_LIST[@]}"; do
            for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
              EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

              echo ""
              echo "EXPERIMENT $EXPERIMENT_COUNT/$TOTAL_COMBINATIONS"
              echo "  LR=$LEARNING_RATE, Beta=$BETA, Top%=$TOP_PERCENT, BS=$BATCH_SIZE, Loss=$LOSS_TYPE"
              echo "  Reward: Holdout (train on split_A, eval on split_B)"

              TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
              EXPERIMENT_DIR="$MAIN_OUTPUT_DIR/loss_${LOSS_TYPE}_lr${LEARNING_RATE}_beta${BETA}_top${TOP_PERCENT}_bs${BATCH_SIZE}_${TIMESTAMP}"
              mkdir -p "$EXPERIMENT_DIR"

              export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

              CUDA_VISIBLE_DEVICES=$GPU_ID python iterative_dpo_trainer.py \
                  --dataname "$DATANAME" \
                  --total_epochs "$TOTAL_EPOCHS" \
                  --learning_rate "$LEARNING_RATE" \
                  --beta "$BETA" \
                  --top_percent "$TOP_PERCENT" \
                  --batch_size "$BATCH_SIZE" \
                  --n_samples_per_iter "$N_SAMPLES_PER_ITER" \
                  --loss_type "$LOSS_TYPE" \
                  --experiment_dir "$EXPERIMENT_DIR" \
                  --model_type "$MODEL_TYPE" \
                  --reward_type "$REWARD_TYPE" \
                  --round_number 1

              if [ $? -eq 0 ]; then
                  echo "DPO training completed for experiment $EXPERIMENT_COUNT"
              else
                  echo "DPO training failed for experiment $EXPERIMENT_COUNT"
                  continue
              fi

              # Run evaluations
              if [ "$RUN_EVALUATION" = true ]; then
                echo "Running standard evaluation..."
                CUDA_VISIBLE_DEVICES=$GPU_ID python run_all_evaluations.py \
                    --experiment_dir "$EXPERIMENT_DIR" \
                    --dataname "$DATANAME" \
                    --model_type "$MODEL_TYPE" || true

                # Run holdout classifier evaluation (independent split_B)
                SPLIT_INFO="$EXPERIMENT_DIR/pairs_data/split_b_indices.json"
                if [ -f "$SPLIT_INFO" ]; then
                    echo "Running holdout classifier evaluation (independent split_B)..."
                    for syn_file in "$EXPERIMENT_DIR"/synthetic_data/iteration_*/*.csv; do
                        if [ -f "$syn_file" ]; then
                            CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_holdout_classifier.py \
                                --dataname "$DATANAME" \
                                --path "$syn_file" \
                                --split_info_path "$SPLIT_INFO" \
                                --save_path "$EXPERIMENT_DIR/evaluation_results/holdout_$(basename $syn_file .csv).txt" || true
                        fi
                    done
                else
                    echo "WARNING: split_b_indices.json not found, skipping holdout eval"
                fi

                # Run proper MIA evaluation
                echo "Running proper MIA evaluation..."
                for syn_file in "$EXPERIMENT_DIR"/synthetic_data/iteration_*/*.csv; do
                    if [ -f "$syn_file" ]; then
                        CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_mia_proper.py \
                            --dataname "$DATANAME" \
                            --path "$syn_file" \
                            --save_path "$EXPERIMENT_DIR/evaluation_results/mia_proper_$(basename $syn_file .csv).txt" || true
                    fi
                done

                # Run statistical distance evaluation
                echo "Running statistical distance evaluation..."
                for syn_file in "$EXPERIMENT_DIR"/synthetic_data/iteration_*/*.csv; do
                    if [ -f "$syn_file" ]; then
                        CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_statistical_distances.py \
                            --dataname "$DATANAME" \
                            --path "$syn_file" \
                            --save_path "$EXPERIMENT_DIR/evaluation_results/statistical_$(basename $syn_file .csv).txt" || true
                    fi
                done
              fi

              # Save experiment summary
              cat > "$EXPERIMENT_DIR/experiment_summary.txt" << EOF
Experiment Summary (Holdout Reward - Option 3):
Reward Type: Holdout (RF on split_A, independent eval on split_B)
Model Type: $MODEL_TYPE
Dataset: $DATANAME
Epochs: $TOTAL_EPOCHS
Learning Rate: $LEARNING_RATE
Beta: $BETA
Top Percent: $TOP_PERCENT
Batch Size: $BATCH_SIZE
Loss Type: $LOSS_TYPE
Samples per Iteration: $N_SAMPLES_PER_ITER
Timestamp: $TIMESTAMP
EOF

              echo "Experiment $EXPERIMENT_COUNT/$TOTAL_COMBINATIONS completed"
              echo "Results: $EXPERIMENT_DIR"

              # GPU cleanup
              if [ $EXPERIMENT_COUNT -lt $TOTAL_COMBINATIONS ]; then
                  CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
"
                  sleep 2
              fi
            done
          done
        done
      done
    done

    echo ""
    echo "ALL EXPERIMENTS COMPLETED FOR $DATANAME"
    echo "Results: $MAIN_OUTPUT_DIR/"

done

echo ""
echo "HOLDOUT REWARD PIPELINE COMPLETED"
echo "Reward: RF classifier on split_A (50% of real data)"
echo "Evaluation: Independent XGBoost on split_B (other 50%)"
