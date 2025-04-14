import mteb
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model_name = "retro_7neg_filtered_retrieval_2_farsi" # 0
# model_name = "retro_9neg_instruct_stage3_with_inbatch_v2" # 2
# model_name = "retro_9neg_instruct_stage3_v2_v4_with_inbatch" # 6
# model_name = "retro_9neg_instruct_stage3_v2_v3_with_inbatch" # 4
# model_name = "retro_9neg_instruct_stage3_v2_v5_with_inbatch" # 01

# model_name = "retro_7neg_filtered_retrieval_2_farsi_instruct" # 1
# model_name = "retro_9neg_instruct_stage3_with_inbatch_v2_instruct" # 3
# model_name = "retro_9neg_instruct_stage3_v2_v4_with_inbatch_instruct" # 7
# model_name = "retro_9neg_instruct_stage3_v2_v3_with_inbatch_instruct" # 5 
# model_name = "retro_9neg_instruct_stage3_v2_v5_with_inbatch_instruct" # 11

# model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_9neg_instruct_stage3_v2_v4_with_inbatch"
# model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_7neg_ourspacedwordpiece_filtered_retrieval_2_farsi"
# model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_7neg_filtered_retrieval_2_farsi"
# model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/RetroMAE_ourspacedwordpiece_ourlinebyline_filtered_retrieval_2_farsi"
# model_name = "retro_rope_balanced_9neg_instruct_stage3_v2_v4_with_inbatch"
# model_name = "bge_large_ret2_9neg_instruct_stage3_v2_v4_with_inbatch"
# model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# model_name = "intfloat/e5-mistral-7b-instruct"
# model_name = "retro_ourwordpiece_retrieval_2_instruct_stage3_v2_v3_with_inbatch"
# model_name = "retro_ourwordpiece_retrieval_2_instruct_stage3_v2_v3_plus_with_inbatch"
# model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/RetroMAE_ourspacedwordpiece_ourlinebyline_filtered_balance_small_farsi2/"
model_name = "retro_ourwordpiece_balance_small_instruct_stage3_v2_v3_plus_with_inbatch"
model_name = "/mnt/data/mehran-workspace/clean-code/base-model-evaluation/models_to_test/checkpoint-3930351-allfarsi_ourdataset_spacedDataset50kWordpiece_mlm_ourlinebylinels"
model_name = "/mnt/data/mehran-workspace/clean-code/base-model-evaluation/models_to_test/checkpoint-1564218-RetroMAE_ourspacedwordpiece_ourlinebyline"
# tasks = mteb.get_benchmark("MTEB(fas, beta)")
datasets = [
    "SynPerQARetrieval",
    "SynPerChatbotTopicsRetrieval",
    "SynPerChatbotRAGTopicsRetrieval",
    "SynPerChatbotRAGFAQRetrieval",
    "PersianWebDocumentRetrieval"
]

tasks = mteb.get_tasks(tasks=datasets, languages=['fas'])
# tasks = ["SynPerChatbotConvSAClassification", "CQADupstackRetrievalFa"]
# print(tasks.tasks)

# model_names_and_tasks = [
#     # ("retro_9neg_instruct_stage3_v2_v3_with_inbatch_instruct", ["Farsick", "Query2Query", "SAMSumFa", "SynPerChatbotRAGSumSRetrieval", "SynPerChatbotSumSRetrieval", "SynPerSTS"]),
#     # ("retro_9neg_instruct_stage3_with_inbatch_v2_instruct", ["Farsick", "Query2Query", "SAMSumFa", "SynPerChatbotRAGSumSRetrieval", "SynPerChatbotSumSRetrieval", "SynPerSTS"]),
#     # ("retro_9neg_instruct_stage3_v2_v4_with_inbatch_instruct", ["Farsick", "Query2Query", "SAMSumFa", "SynPerChatbotRAGSumSRetrieval", "SynPerChatbotSumSRetrieval", "SynPerSTS"]),
#     ("retro_7neg_filtered_retrieval_2_farsi_instruct", ["Farsick", "Query2Query", "SAMSumFa", "SynPerChatbotRAGSumSRetrieval", "SynPerChatbotSumSRetrieval", "SynPerSTS"]),
#]

# for model_name, tasks in model_names_and_tasks:

print(f"\n\nModel Name: {model_name}")

model = mteb.models.get_model(model_name)
evaluation = mteb.MTEB(tasks=tasks)

output_folder_name = "_".join(model_name.split('/'))

results = evaluation.run(model, output_folder=f"test-our-models/{output_folder_name}", verbosity=3, batch_size=128)
