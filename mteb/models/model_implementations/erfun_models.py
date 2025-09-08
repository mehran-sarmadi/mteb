from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)

# Dataset task mappings with descriptions and task IDs
DATASET_TASKS = {
    "PersianTextEmotion": ("دسته بندی , دسته بندی احساس متن", 1),
    "PersianTextEmotion.v2": ("دسته بندی , دسته بندی احساس متن", 1),
    "PersianFoodSentimentClassification": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "SentimentDKSF": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "MassiveIntentClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "MassiveScenarioClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "StyleClassification": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerChatbotConvSAAnger": (
        "دسته بندی , تحلیل احساس عصبانیت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASatisfaction": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAFriendship": (
        "دسته بندی , تحلیل احساس صمیمیت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAFear": (
        "دسته بندی , تحلیل احساس ترس کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAJealousy": (
        "دسته بندی , تحلیل احساس حسادت کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASurprise": (
        "دسته بندی , تحلیل احساس شگفتی کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSALove": (
        "دسته بندی , تحلیل احساس عشق کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSASadness": (
        "دسته بندی , تحلیل احساس غصه کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAHappiness": (
        "دسته بندی , تحلیل احساس خوشحالی کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotConvSAToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotConvSAToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "PersianTextTone": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerTextToneClassification.v3": ("دسته بندی , تشخیص لحن متن", 1),
    "SynPerChatbotToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotRAGToneUserClassification": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        1,
    ),
    "SynPerChatbotRAGToneChatbotClassification": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        1,
    ),
    "SynPerChatbotSatisfactionLevelClassification": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        1,
    ),
    "DigikalamagClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClassification.v2": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClassification": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClassification.v2": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "DeepSentiPers": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "DeepSentiPers.v2": ("دسته بندی , تحلیل احساس رضایت متن", 1),
    "Farsick": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "Query2Query": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "SynPerSTS": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", 3),
    "BeytooteClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "DigikalamagClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "NLPTwitterAnalysisClustering": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "HamshahriClustring": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "SIDClustring": ("دسته بندی , دسته بندی موضوعی متن", 1),
    "MIRACLReranking": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", 3),
    "WikipediaRerankingMultilingual": (
        "تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟",
        3,
    ),
    "SAMSumFa": (
        "تشخیص ارتباط , متن اول یک مکالمه است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotSumSRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotRAGSumSRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        3,
    ),
    "SynPerQARetrieval": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", 3),
    "SynPerChatbotTopicsRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotRAGTopicsRetrieval": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        3,
    ),
    "SynPerChatbotRAGFAQRetrieval": (
        "تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟",
        3,
    ),
    "PersianWebDocumentRetrieval": (
        "تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟",
        3,
    ),
}

# Add all retrieval datasets with the same instruction and task ID
RETRIEVAL_DATASETS = [
    "ArguAna-Fa",
    "ClimateFEVER-Fa",
    "CQADupstackAndroidRetrieval-Fa",
    "CQADupstackEnglishRetrieval-Fa",
    "CQADupstackGamingRetrieval-Fa",
    "CQADupstackGisRetrieval-Fa",
    "CQADupstackMathematicaRetrieval-Fa",
    "CQADupstackPhysicsRetrieval-Fa",
    "CQADupstackProgrammersRetrieval-Fa",
    "CQADupstackStatsRetrieval-Fa",
    "CQADupstackTexRetrieval-Fa",
    "CQADupstackUnixRetrieval-Fa",
    "CQADupstackWebmastersRetrieval-Fa",
    "CQADupstackWordpressRetrieval-Fa",
    "DBPedia-Fa",
    "FiQA2018-Fa",
    "HotpotQA-Fa",
    "MSMARCO-Fa",
    "NFCorpus-Fa",
    "NQ-Fa",
    "QuoraRetrieval-Fa",
    "SCIDOCS-Fa",
    "SciFact-Fa",
    "TRECCOVID-Fa",
    "Touche2020-Fa",
    "MIRACLRetrieval",
    "WikipediaRetrievalMultilingual",
    "MIRACLRetrievalHardNegatives",
    "HotpotQA-FaHardNegatives",
    "MSMARCO-FaHardNegatives",
    "NQ-FaHardNegatives",
    "FEVER-FaHardNegatives",
    "WebFAQRetrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023RetrievalHardNegatives",
    "MIRACLRetrievalHardNegatives",
    "HotpotQA-FaHardNegatives",
    "MSMARCO-FaHardNegatives",
    "NQ-FaHardNegatives",
    "ArguAna-Fa.v2",
    "FiQA2018-Fa.v2",
    "QuoraRetrieval-Fa.v2",
    "SCIDOCS-Fa.v2",
    "SciFact-Fa.v2",
    "TRECCOVID-Fa.v2",
    "Touche2020-Fa.v2",
    "FEVER-FaHardNegatives",
    "NeuCLIR2023RetrievalHardNegatives",
    "WebFAQRetrieval",
]

for dataset in RETRIEVAL_DATASETS:
    DATASET_TASKS[dataset] = ("تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟", 3)


class HakimModelWrapper(Wrapper):
    """A simplified wrapper for the Hakim instruction-following model."""

    def __init__(self, model_name: str, revision: str, **kwargs):
        """Initializes the wrapper and loads the SentenceTransformer model."""
        self.model = SentenceTransformer(model_name, revision=revision)
        # You can still have model_name for other logic if needed
        self.model_name = model_name
        logging.info(f"Initialized model: {model_name}")

    def _preprocess_sample(
        self,
        sample: str,
        task_name: str,
        prompt_type: PromptType | None,
        sub: str | None,
    ) -> str:
        """Preprocesses a single text sample based on the task."""
        if 'Hakim_unsup_lora_retrieval_w_prompt' in self.model_name:
            if prompt_type and prompt_type.value == 'query':
                sample = "متن پرس و جو: " + sample
                return sample

        # if "unsup" in self.model_name:
        #     return sample

        task_prompt, task_id = DATASET_TASKS.get(task_name, (None, None))

        if not task_prompt:
            logger.warning(f"Unknown dataset: {task_name}, no preprocessing applied.")
            return sample

        task_prompt = f"مسئله : {task_prompt}"

        if task_id == 1:
            return f"{task_prompt} | متن : {sample}"
        if task_id == 3:
            if sub == "sentence1" or (prompt_type and prompt_type.value == "query"):
                return f"{task_prompt} | متن اول : {sample}"
            if sub == "sentence2" or (prompt_type and prompt_type.value == "passage"):
                return f"{task_prompt} | متن دوم : {sample}"
        return sample

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes sentences using a loaded SentenceTransformer model.

        Args:
            sentences: A list of strings to be encoded.
            task_name: The name of the task for preprocessing.
            prompt_type: The type of prompt (e.g., 'query', 'passage').
            batch_size: The batch size for encoding.
            **kwargs: Additional keyword arguments.

        Returns:
            A numpy array of the sentence embeddings.
        """
        if not sentences or not all(isinstance(s, str) for s in sentences):
            raise ValueError("Input must be a non-empty list of strings.")

        logger.info(
            f"Starting encoding for {len(sentences)} sentences, task: {task_name}, batch_size: {batch_size}"
        )
        sub = kwargs.get("sub")
        # Pre-process sentences with task-specific instructions if necessary
        print(sentences[0])
        processed_sentences = [
            self._preprocess_sample(s, task_name, prompt_type, sub) for s in sentences
        ]
        print(processed_sentences[0])
        logger.info(f"Encoding {len(processed_sentences)} processed sentences.")

        # Use the sentence-transformers model to encode in batches
        embeddings = self.model.encode(
            processed_sentences,
            batch_size=batch_size,
            show_progress_bar=True,  # Provides a helpful progress bar
            normalize_embeddings=False,  # Set to True if you need unit vectors
        )

        logger.info(f"Encoding completed successfully for {len(embeddings)} sentences.")

        # The output of model.encode is already a numpy array with dtype=np.float32
        return embeddings


############# test with prompt #############
model_1 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_sup_with_lora",
        revision="v1",
    ),
    name="erfun/Hakim_sup_with_lora",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


############# test with prompt #############
model_2 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_unsup_lora_retrieval_w_prompt",
        revision="v1",
    ),
    name="erfun/Hakim_unsup_lora_retrieval_w_prompt",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_3 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_unsup_lora_retrieval_alpha64_r32ـsyn_data",
        revision="v1",
    ),
    name="erfun/Hakim_unsup_lora_retrieval_alpha64_r32ـsyn_data",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

model_4 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_single_gpu",
        revision="v1",
    ),
    name="erfun/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_single_gpu",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

model_5 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_model_parallel/checkpoint-34830",
        revision="v1",
    ),
    name="erfun/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_model_parallel",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_6 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_data_parallel",
        revision="v1",
    ),
    name="erfun/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_data_parallel",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_7 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ret2_data_parallel_lr1e-5_2epoch",
        revision="v1",
    ),
    name="erfun/ret2_data_parallel_lr1e-5_2epoch",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_8 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ret2_data_parallel_lr1e-5_1epoch_long_miracl_syn",
        revision="v1",
    ),
    name="erfun/ret2_data_parallel_lr1e-5_1epoch_long_miracl_syn",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_9 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ret2_data_parallel_lr1e-5_1epoch",
        revision="v1",
    ),
    name="erfun/ret2_data_parallel_lr1e-5_1epoch",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_10 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/hakim_unsup_lora_retrieval_wo_msmarco",
        revision="v1",
    ),
    name="erfun/hakim_unsup_lora_retrieval_wo_msmarco",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_11 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ret2_data_parallel_lr3e-6_1epoch",
        revision="v1",
    ),
    name="erfun/ret2_data_parallel_lr3e-6_1epoch",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_12 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_unsup_lora_retrieval_alpha64_r32",
        revision="v1",
    ),
    name="erfun/Hakim_unsup_lora_retrieval_alpha64_r32",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)


model_13 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_unsup_lora_retrieval",
        revision="v1",
    ),
    name="erfun/Hakim_unsup_lora_retrieval",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)




model_14 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/Hakim_unsup_lora_retrieval_w_prompt",
        revision="v1",
    ),
    name="erfun/Hakim_unsup_lora_retrieval_w_prompt",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

model_15 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_data_parallel_new",
        revision="v1",
    ),
    name="erfun/ourlinebyline8192-lr1e4-1-datav2_retrieval_2_farsi_data_parallel_new",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

model_16 = ModelMeta(
    loader=partial(
        HakimModelWrapper,
        trust_remote_code=True,
        model_name="/mnt/data/ez-workspace/FlagEmbedding_old/FlagEmbedding/baai_general_embedding/results/ret2_data_parallel_lr3e-6_1epoch_long_new",
        revision="v1",
    ),
    name="erfun/ret2_data_parallel_lr3e-6_1epoch_long_new",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-unsup",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)