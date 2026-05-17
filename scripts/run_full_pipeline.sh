#!/bin/bash

# Full DPO Training Pipeline with Versioning Support
# Usage: ./run_full_dpo_pipeline.sh --dataname adult --epochs 3 --lr 5e-4 --beta 10 --top_percent 50 --n_samples_per_iter 5 [--retrain] [--resample] [--rounds=N]

set -e  # Exit on any error


# Default parameters
DATANAME="adult"
TOTAL_EPOCHS=2
LEARNING_RATE="1e-5" 
BETA=1
TOP_PERCENT=100
N_SAMPLES_PER_ITER=3
LOSS_TYPE="dpo"
RETRAIN=false
RESAMPLE=false
ROUNDS=1
RUN_EVALUATION=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataname)
            DATANAME="$2"
            shift 2
            ;;
        --epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --top_percent)
            TOP_PERCENT="$2"
            shift 2
            ;;
        --n_samples_per_iter)
            N_SAMPLES_PER_ITER="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --retrain)
            RETRAIN=true
            shift
            ;;
        --resample)
            RESAMPLE=true
            shift
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --no-eval)
            RUN_EVALUATION=false
            shift
            ;;
        --help|-h)
            echo "🚀 Full DPO Training Pipeline with Versioning Support"
            echo ""
            echo "Usage: $0 --dataname <name> --epochs <num> --lr <rate> [OPTIONS]"
            echo ""
            echo "Required Arguments:"
            echo "  --dataname <name>     Dataset name (adult, diabetes, etc.)"
            echo "  --epochs <num>        Number of DPO training epochs per round"
            echo "  --lr <rate>           Learning rate for DPO training"
            echo ""
            echo "Optional Arguments:"
            echo "  --beta <value>        DPO temperature parameter (default: 10)"
            echo "  --top_percent <num>   Percentage of DPO pairs to use (default: 50)"
            echo "  --n_samples_per_iter <num>     Number of synthetic files per iteration (default: 5)"
            echo "  --loss_type <type>    Loss variant to use (default: dpo)"
            echo "                        Options: dpo, dpo_grad_diff, dpo_kl,"
            echo "                                 tabgraa, tabgraa_logsigmoid, tabgraa_logsigmoid_grad_diff,"
            echo "                                 npo, npo_grad_diff, npo_kl"
            echo "  --rounds <num>        Number of training rounds to run (default: 1)"
            echo ""
            echo "Flags:"
            echo "  --retrain            Force retrain base model even if exists"
            echo "  --resample           Force resample synthetic data even if exists"
            echo "  --no-eval            Skip automatic evaluation (faster but no metrics)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dataname adult --epochs 2 --lr 1e-5"
            echo "  $0 --dataname adult --epochs 3 --lr 5e-4 --beta 15 --rounds 3"
            echo "  $0 --dataname adult --epochs 2 --lr 1e-4 --retrain --resample"
            echo ""
            exit 0
            ;;
        *)
            echo "❌ Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DATANAME" || -z "$TOTAL_EPOCHS" || -z "$LEARNING_RATE" ]]; then
    echo "❌ Missing required parameters!"
    echo "Usage: $0 --dataname <name> --epochs <num> --lr <rate>"
    echo "Use --help for more information"
    exit 1
fi

echo "🚀 Starting Full DPO Training Pipeline with Versioning"
echo "============================================================="
echo "📊 Dataset: $DATANAME"
echo "🔄 Total epochs: $TOTAL_EPOCHS" 
echo "📈 Learning rate: $LEARNING_RATE"
echo "🌡️  Beta: $BETA"
echo "📊 Top percent: $TOP_PERCENT"
echo "📁 Samples per iteration: $N_SAMPLES_PER_ITER"
echo "� Loss type: $LOSS_TYPE"
echo "�🔁 Training rounds: $ROUNDS"
echo "🔄 Retrain models: $RETRAIN"
echo "🎲 Resample data: $RESAMPLE"
echo "============================================================="

# Create output directory structure
OUTPUT_DIR="synthetic/$DATANAME/dpo_experiments"
mkdir -p "$OUTPUT_DIR"

# Main training loop for multiple rounds
for ROUND in $(seq 1 $ROUNDS); do
    echo ""
    echo "🔥 STARTING TRAINING ROUND $ROUND/$ROUNDS"
    echo "============================================================="
    
    # Create experiment timestamp for this round
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_DIR="$OUTPUT_DIR/round_${ROUND}_lr${LEARNING_RATE}_beta${BETA}_top${TOP_PERCENT}_${TIMESTAMP}"
    mkdir -p "$EXPERIMENT_DIR"
    
    echo "📁 Round $ROUND experiment directory: $EXPERIMENT_DIR"
    
    # Step 1: Train base GReaT model (with versioning)
    BASE_MODEL_DIR="baselines/great/ckpt/$DATANAME"
    BASE_MODEL_PATH="$BASE_MODEL_DIR/model_$ROUND.pt"
    ORIGINAL_MODEL_PATH="$BASE_MODEL_DIR/model.pt"
    
    # Check if we need to train/retrain
    # For Round 1: check if original model.pt exists (unless --retrain is used)
    # For Round N>1: check if versioned model exists (unless --retrain is used)
    if [ "$RETRAIN" = true ]; then
        echo ""
        echo "🏋️ Step 1: Training base GReaT model (Round $ROUND) - Force retrain..."
        
        # Train the model (always creates model.pt)
        python main.py --dataname "$DATANAME" --method great --mode train --bs 4
        
        # Copy to versioned model
        if [ -f "$ORIGINAL_MODEL_PATH" ]; then
            cp "$ORIGINAL_MODEL_PATH" "$BASE_MODEL_PATH"
            echo "✅ Base model training completed and saved as: $BASE_MODEL_PATH"
        else
            echo "❌ Training failed - model.pt not found"
            exit 1
        fi
    elif [ "$ROUND" -eq 1 ] && [ -f "$ORIGINAL_MODEL_PATH" ]; then
        echo "✅ Base model already exists: $ORIGINAL_MODEL_PATH"
        # Copy to versioned model for consistency
        cp "$ORIGINAL_MODEL_PATH" "$BASE_MODEL_PATH"
        echo "✅ Saved as versioned model: $BASE_MODEL_PATH"
    elif [ "$ROUND" -gt 1 ] && [ -f "$BASE_MODEL_PATH" ]; then
        echo "✅ Versioned model already exists for round $ROUND: $BASE_MODEL_PATH"
        # Copy versioned model back to model.pt for use
        cp "$BASE_MODEL_PATH" "$ORIGINAL_MODEL_PATH"
    else
        echo ""
        echo "🏋️ Step 1: Training base GReaT model (Round $ROUND)..."
        
        # Train the model (always creates model.pt)
        python main.py --dataname "$DATANAME" --method great --mode train --bs 4
        
        # Copy to versioned model
        if [ -f "$ORIGINAL_MODEL_PATH" ]; then
            cp "$ORIGINAL_MODEL_PATH" "$BASE_MODEL_PATH"
            echo "✅ Base model training completed and saved as: $BASE_MODEL_PATH"
        else
            echo "❌ Training failed - model.pt not found"
            exit 1
        fi
    fi
    
    # Step 2: Generate initial synthetic data (with versioning)
    SYNTHETIC_DIR="synthetic/$DATANAME"
    VERSIONED_SYNTHETIC_PATH="$SYNTHETIC_DIR/great_$ROUND.csv"
    ORIGINAL_SYNTHETIC_PATH="$SYNTHETIC_DIR/great.csv"
    
    # Check if we need to sample/resample
    # For Round 1: check if original great.csv exists (unless --resample is used)
    # For Round N>1: check if versioned data exists (unless --resample is used)
    if [ "$RESAMPLE" = true ]; then
        echo ""
        echo "🎲 Step 2: Generating initial synthetic data (Round $ROUND) - Force resample..."
        
        # Generate synthetic data (always creates great.csv)
        python main.py --dataname "$DATANAME" --method great --mode sample
        
        # Copy to versioned synthetic data
        if [ -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
            cp "$ORIGINAL_SYNTHETIC_PATH" "$VERSIONED_SYNTHETIC_PATH"
            echo "✅ Synthetic data generation completed and saved as: $VERSIONED_SYNTHETIC_PATH"
        else
            echo "❌ Sampling failed - great.csv not found"
            exit 1
        fi
    elif [ "$ROUND" -eq 1 ] && [ -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
        echo "✅ Initial synthetic data already exists: $ORIGINAL_SYNTHETIC_PATH"
        # Copy to versioned data for consistency
        cp "$ORIGINAL_SYNTHETIC_PATH" "$VERSIONED_SYNTHETIC_PATH"
        echo "✅ Saved as versioned data: $VERSIONED_SYNTHETIC_PATH"
    elif [ "$ROUND" -gt 1 ] && [ -f "$VERSIONED_SYNTHETIC_PATH" ]; then
        echo "✅ Versioned synthetic data already exists for round $ROUND: $VERSIONED_SYNTHETIC_PATH"
        # Copy versioned data back to great.csv for use
        cp "$VERSIONED_SYNTHETIC_PATH" "$ORIGINAL_SYNTHETIC_PATH"
    else
        echo ""
        echo "🎲 Step 2: Generating initial synthetic data (Round $ROUND)..."
        
        # Generate synthetic data (always creates great.csv)
        python main.py --dataname "$DATANAME" --method great --mode sample
        
        # Copy to versioned synthetic data
        if [ -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
            cp "$ORIGINAL_SYNTHETIC_PATH" "$VERSIONED_SYNTHETIC_PATH"
            echo "✅ Synthetic data generation completed and saved as: $VERSIONED_SYNTHETIC_PATH"
        else
            echo "❌ Sampling failed - great.csv not found"
            exit 1
        fi
    fi
    
    # Step 3: Run iterative DPO training with enhanced storage
    echo ""
    echo "🔄 Step 3: Running iterative DPO training (Round $ROUND)..."
    python iterative_dpo_trainer.py \
        --dataname "$DATANAME" \
        --total_epochs "$TOTAL_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --beta "$BETA" \
        --top_percent "$TOP_PERCENT" \
        --n_samples_per_iter "$N_SAMPLES_PER_ITER" \
        --loss_type "$LOSS_TYPE" \
        --experiment_dir "$EXPERIMENT_DIR" \
        --round_number "$ROUND"
    
    # Step 4: Run comprehensive evaluation (optional)
    if [ "$RUN_EVALUATION" = true ]; then
        echo ""
        echo "📊 Step 4: Running comprehensive evaluation (Round $ROUND)..."
        python run_all_evaluations.py \
            --experiment_dir "$EXPERIMENT_DIR" \
            --dataname "$DATANAME"
        
        if [ $? -eq 0 ]; then
            echo "✅ Evaluation completed successfully!"
        else
            echo "⚠️ Evaluation encountered some issues, but continuing..."
        fi
    else
        echo ""
        echo "⏭️ Skipping evaluation (--no-eval flag used)"
    fi
    
    echo ""
    echo "✅ ROUND $ROUND COMPLETED!"
    echo "📁 Results saved in: $EXPERIMENT_DIR"
    
    # Save round summary
    echo "Round $ROUND Summary:" > "$EXPERIMENT_DIR/round_summary.txt"
    echo "Dataset: $DATANAME" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Epochs: $TOTAL_EPOCHS" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Learning Rate: $LEARNING_RATE" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Beta: $BETA" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Top Percent: $TOP_PERCENT" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Samples per Iteration: $N_SAMPLES_PER_ITER" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Base Model: $BASE_MODEL_PATH" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Initial Synthetic: $VERSIONED_SYNTHETIC_PATH" >> "$EXPERIMENT_DIR/round_summary.txt"
    echo "Timestamp: $TIMESTAMP" >> "$EXPERIMENT_DIR/round_summary.txt"
    
done

echo ""
echo "🎉 ALL $ROUNDS ROUNDS COMPLETED!"
echo "============================================================="
echo "📊 Training Summary:"
echo "  - Dataset: $DATANAME"
echo "  - Total rounds: $ROUNDS"
echo "  - Epochs per round: $TOTAL_EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Beta: $BETA"
echo "  - Top percent: $TOP_PERCENT"
echo ""
echo "📁 All results saved in: $OUTPUT_DIR/"
echo ""
echo "📋 Generated files per round:"
echo "  - Models: baselines/great/ckpt/$DATANAME/model_[1-$ROUNDS].pt"
echo "  - Initial synthetic data: synthetic/$DATANAME/great_[1-$ROUNDS].csv"
echo "  - Experiment results: $OUTPUT_DIR/round_[1-$ROUNDS]_*/"
echo "  - Evaluation summaries: $OUTPUT_DIR/round_[1-$ROUNDS]_*/evaluation_results/"
echo ""
echo "📈 Evaluation Results (CSV summaries):"
echo "  - Detection: $OUTPUT_DIR/round_*/evaluation_results/detection_summary.csv"
echo "  - Density: $OUTPUT_DIR/round_*/evaluation_results/density_summary.csv"
echo "  - Privacy: $OUTPUT_DIR/round_*/evaluation_results/privacy_summary.csv"
echo "  - MLE: $OUTPUT_DIR/round_*/evaluation_results/mle_summary.csv"

echo ""
echo "💡 To view evaluation results for a specific round:"
echo "  cd $OUTPUT_DIR/round_1_*/evaluation_results && ls *.csv"

echo ""