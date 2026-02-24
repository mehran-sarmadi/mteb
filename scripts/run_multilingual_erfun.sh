#!/bin/bash
# Run the last 3 erfun multilingual models on MTEB(Multilingual, v2)
# Each model on a separate GPU

BENCHMARK="MTEB(Multilingual, v2)"
OUTPUT_DIR="results"
LOG_DIR="logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

declare -A MODELS
MODELS[0]="erfun/hakim_multi_v1_3"
MODELS[1]="erfun/hakim_multi_v1_3epoch_ch_2000000"
MODELS[2]="erfun/hakim_multi_v1_3epoch_ch_1000000"

for GPU_ID in 0 1 2; do
    MODEL="${MODELS[$GPU_ID]}"
    # Sanitize model name for log file
    LOG_NAME=$(echo "$MODEL" | tr '/' '_')

    echo "Starting $MODEL on GPU $GPU_ID ..."
    CUDA_VISIBLE_DEVICES=$GPU_ID mteb run \
        -m "$MODEL" \
        -b "$BENCHMARK" \
        --output-folder "$OUTPUT_DIR" \
        > "$LOG_DIR/${LOG_NAME}.log" 2>&1 &

    echo "  PID: $!  |  Log: $LOG_DIR/${LOG_NAME}.log"
done

echo ""
echo "All 3 models launched. Monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Wait for all to finish:"
echo "  wait"
wait
echo "All done."
