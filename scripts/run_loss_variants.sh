#!/bin/bash

# Run TabDPO Pipeline for All Loss Types with Custom Folder Name
# This script runs DPO training for all supported loss variants with customizable model and parameters
# Usage: ./run_custom_loss_types.sh --folder_name <custom_name> --model <model_type> [OPTIONS]

set -e  # Exit on any error

# Export NCCL environment variables for RTX 4000 series compatibility
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"  

# Default parameters
DATANAME_LIST=("adult"  )  # Can be overridden to run multiple datasets
MODEL_TYPE="great"  # Default model type: "great" or "great_gemma"
TOTAL_EPOCHS=1
LEARNING_RATE_LIST=("1e-7")  # List of learning rates to test
BETA_LIST=(1)               # List of beta values to test
TOP_PERCENT_LIST=(100)       # List of top_percent values to test
BATCH_SIZE_LIST=(4)         # Keep small for 4090 (24GB). Use 2 for great_gemma.
N_SAMPLES_PER_ITER=1
RUN_EVALUATION=true
RETRAIN=false
RESAMPLE=false
CUSTOM_FOLDER_NAME="aba_gpt2-large_1e-7"  # This will be set by user

# GPU device selection - use CUDA_VISIBLE_DEVICES if set, otherwise default to 0
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    GPU_ID="$CUDA_VISIBLE_DEVICES"
else
    GPU_ID="0"
fi

# All supported loss types from iterative_dpo_trainer.py
LOSS_TYPES=(
    # "dpo"
    # "dpo_grad_diff" 
    # "dpo_kl"
    "tabgraa"
    # "tabgraa_logsigmoid"
    # "tabgraa_logsigmoid_grad_diff"
    # "kto_sigmoid"
    # "kto_logsigmoid"
    # "kto_logsigmoid_grad_diff"
    # "npo"
    # "npo_grad_diff"
    # "npo_kl"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --folder_name)
            CUSTOM_FOLDER_NAME="$2"
            shift 2
            ;;
        --model|--model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --dataname)
            IFS=',' read -ra DATANAME_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --datanames)
            IFS=',' read -ra DATANAME_LIST <<< "$2"  # Alternative parameter name
            shift 2
            ;;
        --epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            IFS=',' read -ra LEARNING_RATE_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --lr_list|--learning_rate_list)
            IFS=',' read -ra LEARNING_RATE_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --beta)
            IFS=',' read -ra BETA_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --beta_list)
            IFS=',' read -ra BETA_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --top_percent)
            IFS=',' read -ra TOP_PERCENT_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --top_percent_list)
            IFS=',' read -ra TOP_PERCENT_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --batch_size)
            IFS=',' read -ra BATCH_SIZE_LIST <<< "$2"  # Support comma-separated list
            shift 2
            ;;
        --batch_size_list)
            IFS=',' read -ra BATCH_SIZE_LIST <<< "$2"  # Alternative parameter name
            shift 2
            ;;
        --n_samples_per_iter)
            N_SAMPLES_PER_ITER="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --loss_types)
            IFS=',' read -ra LOSS_TYPES <<< "$2"  # Support custom loss types
            shift 2
            ;;
        --loss_type_list)
            IFS=',' read -ra LOSS_TYPES <<< "$2"  # Alternative parameter name
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --no-eval)
            RUN_EVALUATION=false
            shift
            ;;
        --retrain)
            RETRAIN=true
            shift
            ;;
        --resample)
            RESAMPLE=true
            shift
            ;;
        --help|-h)
            echo "🚀 Run TabDPO Pipeline for All Loss Types with Custom Folder Name"
            echo ""
            echo "This script runs DPO training for multiple loss variants with customizable models."
            echo ""
            echo "Usage: $0 --folder_name <custom_name> --model <model_type> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --folder_name <name>  Custom name for experiment folder"
            echo "  --model <type>        Model type: great or great_gemma (default: great)"
            echo ""
            echo "Options:"
            echo "  --dataname <name>     Dataset name or comma-separated list (default: adult)"
            echo "                        Examples: adult, \"adult,beijing,diabetes\", \"adult,default,shoppers\""
            echo "  --datanames <list>    Alternative way to specify multiple datasets"
            echo "  --epochs <num>        Number of DPO training epochs (default: 2)"
            echo "  --lr <rate>           Learning rate(s) for DPO training (default: 1e-5)"
            echo "                        Single: 1e-5  |  Multiple: \"1e-5,2e-5,5e-5\""
            echo "  --lr_list <rates>     Alternative way to specify learning rate list"
            echo "  --beta <value>        DPO temperature parameter(s) (default: 20)"
            echo "                        Single: 20  |  Multiple: \"10,20,30\""
            echo "  --beta_list <values>  Alternative way to specify beta list"
            echo "  --top_percent <num>   Percentage(s) of DPO pairs to use (default: 100)"
            echo "                        Single: 100  |  Multiple: \"50,75,100\""
            echo "  --top_percent_list <nums>  Alternative way to specify top_percent list"
            echo "  --batch_size <size>   Batch size(s) for training (default: 4)"
            echo "                        Single: 4  |  Multiple: \"4,8,16,32\""
            echo "  --batch_size_list <sizes>  Alternative way to specify batch size list"
            echo "  --n_samples_per_iter <num>  Number of synthetic files per iteration (default: 2)"
            echo "  --loss_types <list>   Comma-separated list of loss types to run (default: all)"
            echo "                        Example: \"dpo,npo,tabgraa\""
            echo "  --gpu_id <id>         GPU device ID to use (default: auto-detect from CUDA_VISIBLE_DEVICES or 0)"
            echo "  --no-eval             Skip automatic evaluation"
            echo "  --retrain             Force retrain base model even if exists"
            echo "  --resample            Force resample synthetic data even if exists"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Output folder will be: synthetic/<dataname>/<folder_name>"
            echo ""
            echo "Supported loss types:"
            for loss_type in "${LOSS_TYPES[@]}"; do
                echo "  - $loss_type"
            done
            echo ""
            echo "Examples:"
            echo "  $0 --folder_name great_experiment --model great"
            echo "  $0 --folder_name gemma_experiment --model great_gemma"
            echo "  $0 --folder_name multi_dataset --model great_gemma --dataname \"adult,beijing,diabetes\""
            echo "  $0 --folder_name hyperopt_run1 --model great --datanames \"adult,default\" --epochs 3"
            echo "  $0 --folder_name beta_sweep --model great_gemma --dataname adult --beta \"5,10,20\" --lr \"1e-5,2e-5\""
            echo "  $0 --folder_name batch_study --model great --batch_size \"4,8,16,32\""
            echo "  $0 --folder_name select_losses --model great --loss_types \"dpo,npo,tabgraa\""
            echo "  $0 --folder_name grid_search --model great_gemma --lr \"1e-6,1e-5\" --beta \"10,20\" --batch_size \"4,8\""
            echo "  $0 --folder_name quick_test --model great_gemma --dataname beijing --epochs 1 --no-eval"
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
if [[ -z "$CUSTOM_FOLDER_NAME" ]]; then
    echo "❌ Missing required parameter: --folder_name"
    echo "Usage: $0 --folder_name <custom_name> --model <model_type> [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

if [[ "$MODEL_TYPE" != "great" ]] && [[ "$MODEL_TYPE" != "great_gemma" ]]; then
    echo "❌ Invalid model type: $MODEL_TYPE"
    echo "Supported model types: great, great_gemma"
    exit 1
fi

echo "🚀 Running TabDPO Pipeline for All Loss Types"
echo "============================================================="
echo "🤖 Model Type: $MODEL_TYPE"
echo "📊 Dataset(s): ${DATANAME_LIST[*]}"
echo "📁 Custom folder: $CUSTOM_FOLDER_NAME"
echo "🔄 Total epochs: $TOTAL_EPOCHS"
echo "📈 Learning rate(s): ${LEARNING_RATE_LIST[*]}"
echo "🌡️  Beta(s): ${BETA_LIST[*]}"
echo "📊 Top percent(s): ${TOP_PERCENT_LIST[*]}"
echo "📦 Batch size(s): ${BATCH_SIZE_LIST[*]}"
echo "📁 Samples per iteration: $N_SAMPLES_PER_ITER"
echo "🖥️  GPU device: $GPU_ID"
echo "🔄 Retrain models: $RETRAIN"
echo "🎲 Resample data: $RESAMPLE"
echo "📊 Run evaluation: $RUN_EVALUATION"
echo ""
echo "🎯 Loss types to run (${#LOSS_TYPES[@]} total):"
for loss_type in "${LOSS_TYPES[@]}"; do
    echo "  - $loss_type"
done

# Calculate total combinations
TOTAL_LR=${#LEARNING_RATE_LIST[@]}
TOTAL_BETA=${#BETA_LIST[@]}  
TOTAL_TOP_PERCENT=${#TOP_PERCENT_LIST[@]}
TOTAL_BATCH_SIZE=${#BATCH_SIZE_LIST[@]}
TOTAL_LOSS_TYPES=${#LOSS_TYPES[@]}
TOTAL_COMBINATIONS=$((TOTAL_LR * TOTAL_BETA * TOTAL_TOP_PERCENT * TOTAL_BATCH_SIZE * TOTAL_LOSS_TYPES))
echo ""
echo "🔢 Parameter combinations per dataset:"
echo "  - Learning rates: $TOTAL_LR"
echo "  - Beta values: $TOTAL_BETA" 
echo "  - Top percents: $TOTAL_TOP_PERCENT"
echo "  - Batch sizes: $TOTAL_BATCH_SIZE"
echo "  - Loss types: $TOTAL_LOSS_TYPES"
echo "  - Total experiments per dataset: $TOTAL_COMBINATIONS"
echo "============================================================="

# Main loop for multiple datasets
TOTAL_DATASETS=${#DATANAME_LIST[@]}
DATASET_COUNT=0

for DATANAME in "${DATANAME_LIST[@]}"; do
    DATASET_COUNT=$((DATASET_COUNT + 1))
    
    echo ""
    echo "🚀 PROCESSING DATASET $DATASET_COUNT/$TOTAL_DATASETS: $DATANAME"
    echo "============================================================="

    # Set model-specific paths
    BASE_MODEL_DIR="baselines/$MODEL_TYPE/ckpt/$DATANAME"
    BASE_MODEL_PATH="$BASE_MODEL_DIR/model.pt"
    SYNTHETIC_DIR="synthetic/$DATANAME"
    ORIGINAL_SYNTHETIC_PATH="$SYNTHETIC_DIR/$MODEL_TYPE.csv"

    # Step 1: Train base model (if needed or forced)
    if [ "$RETRAIN" = true ]; then
        echo ""
        echo "🏋️ Step 1: Training base $MODEL_TYPE model (Force retrain)..."
        
        # Create model directory if it doesn't exist
        mkdir -p "$BASE_MODEL_DIR"
        
        # Train the model
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode train --bs 16
        
        if [ -f "$BASE_MODEL_PATH" ]; then
            echo "✅ Base model training completed: $BASE_MODEL_PATH"
        else
            echo "❌ Training failed - model.pt not found"
            exit 1
        fi
    elif [ ! -f "$BASE_MODEL_PATH" ]; then
        echo ""
        echo "🏋️ Step 1: Training base $MODEL_TYPE model (not found)..."
        
        # Create model directory if it doesn't exist
        mkdir -p "$BASE_MODEL_DIR"
        
        # Train the model
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode train --bs 16
        
        if [ -f "$BASE_MODEL_PATH" ]; then
            echo "✅ Base model training completed: $BASE_MODEL_PATH"
        else
            echo "❌ Training failed - model.pt not found"
            exit 1
        fi
    else
        echo "✅ Base model found: $BASE_MODEL_PATH"
    fi

    # Step 2: Generate initial synthetic data (if needed or forced)
    if [ "$RESAMPLE" = true ]; then
        echo ""
        echo "🎲 Step 2: Generating initial synthetic data (Force resample)..."
        
        # Create synthetic directory if it doesn't exist
        mkdir -p "$SYNTHETIC_DIR"
        
        # Generate synthetic data
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode sample
        
        if [ -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
            echo "✅ Synthetic data generation completed: $ORIGINAL_SYNTHETIC_PATH"
        else
            echo "❌ Sampling failed - $MODEL_TYPE.csv not found"
            exit 1
        fi
    elif [ ! -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
        echo ""
        echo "🎲 Step 2: Generating initial synthetic data (not found)..."
        
        # Create synthetic directory if it doesn't exist
        mkdir -p "$SYNTHETIC_DIR"
        
        # Generate synthetic data
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataname "$DATANAME" --method $MODEL_TYPE --mode sample
        
        if [ -f "$ORIGINAL_SYNTHETIC_PATH" ]; then
            echo "✅ Synthetic data generation completed: $ORIGINAL_SYNTHETIC_PATH"
        else
            echo "❌ Sampling failed - $MODEL_TYPE.csv not found"
            exit 1
        fi
    else
        echo "✅ Initial synthetic data found: $ORIGINAL_SYNTHETIC_PATH"
    fi
    echo ""

    # Create main output directory with custom folder name
    MAIN_OUTPUT_DIR="synthetic/$DATANAME/$CUSTOM_FOLDER_NAME"
    mkdir -p "$MAIN_OUTPUT_DIR"

    echo "📁 Custom output directory created for $DATANAME: $MAIN_OUTPUT_DIR"
    echo ""

    # Counter for progress tracking  
    EXPERIMENT_COUNT=0

    # Initial GPU status check
    echo "🔧 Checking GPU status before starting experiments..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    free = total_mem - allocated
    print(f'GPU: {torch.cuda.get_device_name(device)}')
    print(f'Total GPU Memory: {total_mem:.1f}GB')
    print(f'Allocated: {allocated:.1f}GB')
    print(f'Cached: {cached:.1f}GB')
    print(f'Free: {free:.1f}GB')
else:
    print('CUDA not available')
"
    echo ""

    # Nested loops for parameter grid search
    for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
      for LEARNING_RATE in "${LEARNING_RATE_LIST[@]}"; do
        for BETA in "${BETA_LIST[@]}"; do
          for TOP_PERCENT in "${TOP_PERCENT_LIST[@]}"; do
            for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
              EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
              
              echo ""
              echo "🔥 RUNNING EXPERIMENT $EXPERIMENT_COUNT/$TOTAL_COMBINATIONS"
              echo "🤖 Model: $MODEL_TYPE | Dataset: $DATANAME"
              echo "🎯 Parameters: LR=${LEARNING_RATE}, Beta=${BETA}, Top%=${TOP_PERCENT}, Batch=${BATCH_SIZE}, Loss=${LOSS_TYPE}"
              echo "============================================================="
              
              # Create timestamp for this experiment
              TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
              EXPERIMENT_DIR="$MAIN_OUTPUT_DIR/loss_${LOSS_TYPE}_lr${LEARNING_RATE}_beta${BETA}_top${TOP_PERCENT}_bs${BATCH_SIZE}_${TIMESTAMP}"
              mkdir -p "$EXPERIMENT_DIR"
              
              echo "📁 Experiment directory: $EXPERIMENT_DIR"
              
              # GPU memory check before training
              echo "🔧 GPU memory status before training:"
              CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free = total_mem - allocated
    print(f'  Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Free: {free:.1f}GB')
"
              
              # Run the DPO training with current parameter combination
              echo "🔄 Running iterative DPO training..."
              
              # Set PyTorch memory management
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
                  --round_number 1
              
              if [ $? -eq 0 ]; then
                  echo "✅ DPO training completed successfully for experiment $EXPERIMENT_COUNT"
              else
                  echo "❌ DPO training failed for experiment $EXPERIMENT_COUNT"
                  echo "Continuing with next experiment..."
                  continue
              fi
              
              # Run evaluation if enabled
              if [ "$RUN_EVALUATION" = true ]; then
                echo "📊 Running evaluation..."
                CUDA_VISIBLE_DEVICES=0 python run_all_evaluations.py \
                    --experiment_dir "$EXPERIMENT_DIR" \
                    --dataname "$DATANAME" \
                    --model_type "$MODEL_TYPE"
                  
                  if [ $? -eq 0 ]; then
                      echo "✅ Evaluation completed successfully for experiment $EXPERIMENT_COUNT"
                  else
                      echo "⚠️ Evaluation encountered issues for experiment $EXPERIMENT_COUNT, but continuing..."
                  fi
              else
                  echo "⏭️ Skipping evaluation for experiment $EXPERIMENT_COUNT"
              fi
              
              # Save experiment summary with all parameters
              cat > "$EXPERIMENT_DIR/experiment_summary.txt" << EOF
Loss Type Experiment Summary:
Model Type: $MODEL_TYPE
Custom Folder: $CUSTOM_FOLDER_NAME
Dataset: $DATANAME
Epochs: $TOTAL_EPOCHS
Learning Rate: $LEARNING_RATE
Beta: $BETA
Top Percent: $TOP_PERCENT
Batch Size: $BATCH_SIZE
Loss Type: $LOSS_TYPE
Samples per Iteration: $N_SAMPLES_PER_ITER
Timestamp: $TIMESTAMP
Experiment Number: $EXPERIMENT_COUNT
Total Combinations: $TOTAL_COMBINATIONS
EOF
              
              echo "✅ EXPERIMENT $EXPERIMENT_COUNT/$TOTAL_COMBINATIONS COMPLETED!"
              echo "📁 Results saved in: $EXPERIMENT_DIR"
              
              # GPU cleanup before next experiment (if not the last one)
              if [ $EXPERIMENT_COUNT -lt $TOTAL_COMBINATIONS ]; then
                  echo ""
                  echo "🧹 Cleaning up GPU memory before next experiment..."
                  
                  # Force Python garbage collection and GPU memory cleanup
                  CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'GPU memory freed. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
gc.collect()
print('GPU cleanup completed')
"
                  
                  # Small pause to ensure cleanup completes
                  sleep 2
                  echo "✅ GPU cleanup completed"
                  echo ""
              fi
            done  # End loss type loop
          done    # End top_percent loop
        done      # End beta loop
      done        # End learning_rate loop
    done          # End batch_size loop

    echo ""
    echo "🎉 ALL EXPERIMENTS COMPLETED FOR $DATANAME!"
    echo "============================================================="
    echo "📊 Training Summary:"
    echo "  - Model: $MODEL_TYPE"
    echo "  - Dataset: $DATANAME"
    echo "  - Custom folder: $CUSTOM_FOLDER_NAME"
    echo "  - Total experiments run: $EXPERIMENT_COUNT"
    echo "  - Learning rates tested: ${LEARNING_RATE_LIST[*]}"
    echo "  - Beta values tested: ${BETA_LIST[*]}"
    echo "  - Top percents tested: ${TOP_PERCENT_LIST[*]}"
    echo "  - Batch sizes tested: ${BATCH_SIZE_LIST[*]}"
    echo "  - Loss types tested: ${LOSS_TYPES[*]}"
    echo "  - Epochs per experiment: $TOTAL_EPOCHS"
    echo "  - Retrain model: $RETRAIN"
    echo "  - Resample data: $RESAMPLE"
    echo ""
    echo "📁 All results saved in: $MAIN_OUTPUT_DIR/"
    echo "   Full path: $(pwd)/$MAIN_OUTPUT_DIR/"
    echo ""
    echo "📋 Generated directories per parameter combination:"
    echo "     Format: loss_<TYPE>_lr<RATE>_beta<VALUE>_top<PERCENT>_bs<SIZE>_<TIMESTAMP>/"
    echo ""
    echo "💡 To view results for a specific experiment:"
    echo "  cd $MAIN_OUTPUT_DIR/loss_<LOSS_TYPE>_lr<LR>_beta<BETA>_top<TOP_PERCENT>_bs<BATCH_SIZE>_*/evaluation_results && ls *.csv"
    echo ""
    echo "📈 To compare results across all parameter combinations:"
    echo "  find $MAIN_OUTPUT_DIR -name '*.csv' | grep evaluation_results"
    echo ""

    # Create a summary script for easy result comparison
    SUMMARY_SCRIPT="$MAIN_OUTPUT_DIR/compare_results.sh"
    cat > "$SUMMARY_SCRIPT" << 'EOF'
#!/bin/bash
# Summary script to compare results across all loss types

echo "📊 Comparing Results Across All Loss Types"
echo "============================================"

for dir in loss_*/; do
    if [ -d "$dir" ]; then
        loss_type=$(echo "$dir" | sed 's/loss_\(.*\)_lr.*/\1/')
        echo ""
        echo "🔍 Loss Type: $loss_type"
        echo "   Directory: $dir"
        
        # Check for evaluation results
        if [ -d "$dir/evaluation_results" ]; then
            echo "   📈 Evaluation files:"
            find "$dir/evaluation_results" -name "*.csv" | sed 's/^/      /'
        else
            echo "   ❌ No evaluation results found"
        fi
    fi
done

echo ""
echo "💡 To extract specific metrics, use:"
echo "   grep 'metric_name' */evaluation_results/*.csv"
EOF

    chmod +x "$SUMMARY_SCRIPT"
    echo "📋 Created comparison script for $DATANAME: $SUMMARY_SCRIPT"
    echo "   Run with: cd $MAIN_OUTPUT_DIR && ./compare_results.sh"

    echo ""
    echo "✅ DATASET $DATANAME COMPLETED! ($DATASET_COUNT/$TOTAL_DATASETS)"
    echo "📁 Results for $DATANAME saved in: $MAIN_OUTPUT_DIR/"
    echo "============================================================="

done  # End of dataset loop

echo ""
echo "🎉 ALL DATASETS AND LOSS TYPES COMPLETED!"
echo "============================================================="
echo "📊 Final Summary:"
echo "  - Model Type: $MODEL_TYPE"
echo "  - Datasets processed: ${DATANAME_LIST[*]}"
echo "  - Custom folder: $CUSTOM_FOLDER_NAME"
echo "  - Learning rates tested: ${LEARNING_RATE_LIST[*]}"
echo "  - Beta values tested: ${BETA_LIST[*]}"
echo "  - Top percents tested: ${TOP_PERCENT_LIST[*]}"
echo "  - Batch sizes tested: ${BATCH_SIZE_LIST[*]}"
echo "  - Loss types tested: ${LOSS_TYPES[*]}"
echo "  - Epochs per experiment: $TOTAL_EPOCHS"
echo "  - Retrain model: $RETRAIN"
echo "  - Resample data: $RESAMPLE"
echo ""
echo "📁 Results saved for each dataset:"
for DATANAME in "${DATANAME_LIST[@]}"; do
    echo "  - $DATANAME: $(pwd)/synthetic/$DATANAME/$CUSTOM_FOLDER_NAME/"
done
echo ""
echo "📋 Generated directories per dataset and parameter combination:"
for DATANAME in "${DATANAME_LIST[@]}"; do
    echo "  Dataset: $DATANAME"
    echo "    Format: loss_<TYPE>_lr<RATE>_beta<VALUE>_top<PERCENT>_bs<SIZE>_<TIMESTAMP>/"
done
echo ""
echo "💡 Quick commands to analyze results:"
echo "  # View all evaluation CSVs:"
echo "  find synthetic/*/$CUSTOM_FOLDER_NAME -name '*.csv' -path '*/evaluation_results/*' | xargs ls -la"
echo ""
echo "  # Extract specific metrics across all experiments:"
echo "  grep -r 'correlation' synthetic/*/$CUSTOM_FOLDER_NAME/*/evaluation_results/*.csv"
echo "  grep -r 'accuracy' synthetic/*/$CUSTOM_FOLDER_NAME/*/evaluation_results/*.csv"
echo ""
echo "  # Compare batch size effects:"
echo "  grep -r 'loss' synthetic/*/$CUSTOM_FOLDER_NAME/*/experiment_summary.txt | grep -E '(batch|Batch)'"
echo ""
echo "📈 Batch Size Study Analysis Tips:"
echo "  - Compare same loss types with different batch sizes"
echo "  - Look for stability differences in results"
echo "  - Check if larger batches improve KTO performance (reduces mean variance)"
echo "  - Note memory usage differences in training logs"
echo ""
echo "📈 Model Comparison Tips:"
echo "  - For GReaT model results: look in synthetic/*/great_*/"
echo "  - For Gemma model results: look in synthetic/*/great_gemma_*/"
echo "  - Compare same loss types across different models"
echo ""
echo "✅ Script execution completed successfully!"