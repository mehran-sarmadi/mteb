#!/bin/bash
# Run all test models in parallel on separate GPUs against FaMTEB v2

cd "$(dirname "$0")"

# Ensure peft is installed for LoRA model
uv pip install peft 2>/dev/null

BENCHMARK="MTEB(fas, v2)"
OUTPUT_BASE="results"

# GPU 0: test_embedding_model
CUDA_VISIBLE_DEVICES=0 uv run mteb run \
    -m "MCINext/test-embedding-model" \
    -b "$BENCHMARK" \
    --output-folder "$OUTPUT_BASE/test-embedding-model" \
    2>&1 | tee logs/test-embedding-model.log &

# GPU 1: test_embedding_model_lora
CUDA_VISIBLE_DEVICES=1 uv run mteb run \
    -m "MCINext/test-embedding-model-lora" \
    -b "$BENCHMARK" \
    --output-folder "$OUTPUT_BASE/test-embedding-model-lora" \
    2>&1 | tee logs/test-embedding-model-lora.log &

# GPU 2-6: test_embedding_model_matryoshka at each dimension
DIMS=(768 512 256 128 64)
for i in "${!DIMS[@]}"; do
    GPU=$((i + 2))
    DIM=${DIMS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU uv run python -c "
import mteb

model = mteb.get_model('MCINext/test-embedding-model-matryoshka', embed_dim=$DIM)
tasks = mteb.get_benchmarks(names=['$BENCHMARK'])[0].tasks
mteb.evaluate(
    model,
    tasks,
    cache=mteb.ResultCache('$OUTPUT_BASE/test-embedding-model-matryoshka-dim$DIM'),
)
" 2>&1 | tee "logs/test-embedding-model-matryoshka-dim$DIM.log" &
done

echo "All 7 jobs launched on GPUs 0-6. Monitoring..."
wait
echo "All jobs completed."
