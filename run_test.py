from __future__ import annotations

import os

import mteb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "MCINext/Hakim"
tasks = mteb.get_benchmark("MTEB(fas, v1)")

print(f"\n\nModel Name: {model_name}")

model = mteb.models.get_model(model_name)
evaluation = mteb.MTEB(tasks=tasks)

output_folder_name = "_".join(model_name.split("/"))

results = evaluation.run(
    model,
    output_folder=f"test-our-models/{output_folder_name}",
    verbosity=3,
    batch_size=128,
)
