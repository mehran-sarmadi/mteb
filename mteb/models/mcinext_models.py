from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Any

import numpy as np
import requests
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task configurations
TASK_CONFIGS = {
    "1_1": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        ["عالی", "خوب", "متوسط", "بد", "خیلی بد"],
    ),
    "1_2": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    ),
    "1_3": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    ),
    "1_4": (
        "دسته بندی , تحلیل احساس عصبانیت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_5": (
        "دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_6": (
        "دسته بندی , تحلیل احساس صمیمیت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_7": ("دسته بندی , تحلیل احساس ترس کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_8": (
        "دسته بندی , تحلیل احساس حسادت کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_9": (
        "دسته بندی , تحلیل احساس شگفتی کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_10": ("دسته بندی , تحلیل احساس عشق کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_11": ("دسته بندی , تحلیل احساس غصه کاربر در مکالمه با چت بات", ["مثبت", "منفی"]),
    "1_12": (
        "دسته بندی , تحلیل احساس خوشحالی کاربر در مکالمه با چت بات",
        ["مثبت", "منفی"],
    ),
    "1_13": ("دسته بندی , تشخیص لحن متن", ["عامیانه", "رسمی", "کودکانه", "ادبی"]),
    "1_14": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "بازی ویدیویی",
            "راهنمای خرید",
            "سلامت و زیبایی",
            "علم و تکنولوژی",
            "عمومی",
            "هنر و سینما",
            "کتاب و ادبیات",
        ],
    ),
    "1_15": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "پزشکی",
            "کشاورزی و منابع طبیعی",
            "فنی مهندسی",
            "علوم پایه",
            "علوم انسانی",
            "هنر و معماری",
            "علمی تخصصی",
            "دامپزشکی",
        ],
    ),
    "1_16": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "هنر و طراحی",
            "مسائل اجتماعی و فعال‌سازی",
            "الهام‌بخش و انگیزشی",
            "خودرو",
            "زیبایی و لوازم آرایشی",
            "غذا و آشپزی",
            "کسب و کار و مالی",
            "مد و سبک",
            "آموزش و یادگیری",
            "علم و کشف",
            "بازی",
            "فناوری و نوآوری",
            "مذهب و معنویت",
            "حیوانات خانگی و جانوران",
            "سفر و ماجراجویی",
            "خانواده و پرورش فرزند",
            "خنده‌دار و طنز",
            "سلامت و بهزیستی",
            "خانه و باغ",
            "سیاست و مسائل روز",
            "تفریحات و فرهنگ عامه",
            "ورزش و ورزشکاری",
            "آب و هوا و فصول",
            "کتاب‌ها و ادبیات",
            "محیط زیست و پایداری",
        ],
    ),
    "1_17": (
        "دسته بندی , دسته بندی موضوعی متن",
        [
            "موسیقی",
            "تقویم",
            "هشدار",
            "آب‌وهوا",
            "پخش",
            "تاریخ و زمان",
            "آشپزی",
            "ایمیل",
            "بیرون‌بر",
            "اخبار",
            "پیشنهاد",
            "فهرست‌ها",
            "اجتماعی",
            "حمل‌ونقل",
            "عمومی",
            "پرسش و پاسخ",
            "صوتی",
            "اینترنت اشیاء",
        ],
    ),
    "1_18": (
        "دسته بندی , دسته بندی احساس متن",
        ["شادی", "غم", "خشم", "انزجار", "ترس", "تعجب"],
    ),
    "1_19": ("دسته بندی , تحلیل احساس رضایت متن", ["مثبت", "منفی", "خنثی"]),
    "1_20": (
        "دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        ["رسمی", "عامیانه", "کودکانه"],
    ),
    "1_21": (
        "دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        ["رسمی", "عامیانه", "کودکانه"],
    ),
    "1_170": ("دسته بندی , دسته بندی موضوعی متن", []),
    "1_190": ("دسته بندی , تحلیل احساس رضایت متن", ["مثبت", "منفی"]),
    "3_1": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        None,
    ),
    "3_5": ("تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟", None),
    "3_6": ("تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟", None),
    "3_12": (
        "تشخیص ارتباط , متن اول یک مکالمه است. آیا متن دوم خلاصه ی متن اول است ؟",
        None,
    ),
    "3_13": ("تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟", None),
    "3_14": (
        "تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        None,
    ),
}

# Dataset task mappings
DATASET_TASKS = {
    "PersianTextEmotion": (1, 18),
    "PersianFoodSentimentClassification": (1, 190),
    "SentimentDKSF": (1, 19),
    "MassiveIntentClassification": (1, 170),
    "MassiveScenarioClassification": (1, 17),
    "SynPerChatbotConvSAAnger": (1, 4),
    "SynPerChatbotConvSASatisfaction": (1, 5),
    "SynPerChatbotConvSAFriendship": (1, 6),
    "SynPerChatbotConvSAFear": (1, 7),
    "SynPerChatbotConvSAJealousy": (1, 8),
    "SynPerChatbotConvSASurprise": (1, 9),
    "SynPerChatbotConvSALove": (1, 10),
    "SynPerChatbotConvSASadness": (1, 11),
    "SynPerChatbotConvSAHappiness": (1, 12),
    "SynPerChatbotConvSAToneChatbotClassification": (1, 21),
    "SynPerChatbotConvSAToneUserClassification": (1, 20),
    "PersianTextTone": (1, 13),
    "SynPerChatbotToneUserClassification": (1, 2),
    "SynPerChatbotToneChatbotClassification": (1, 3),
    "SynPerChatbotRAGToneUserClassification": (1, 2),
    "SynPerChatbotRAGToneChatbotClassification": (1, 3),
    "SynPerChatbotSatisfactionLevelClassification": (1, 1),
    "DigimagClassification": (1, 14),
    "NLPTwitterAnalysisClassification": (1, 16),
    "SIDClassification": (1, 15),
    "DeepSentiPers": (1, 19),
    "DigikalamagClassification": (1, 14),
    "FarsTail": (4, 6),
    "ParsinluEntail": (4, 6),
    "ParsinluQueryParaphPC": (4, 7),
    "SynPerChatbotRAGFAQPC": (4, 1),
    "SynPerTextKeywordsPC": (4, 2),
    "SynPerQAPC": (4, 3),
    "CExaPPC": (4, 7),
    "FarsiParaphraseDetection": (4, 7),
    "Farsick": (3, 6),
    "Query2Query": (3, 6),
    "SynPerSTS": (3, 6),
    "BeytooteClustering": (1, 170),
    "DigikalamagClustering": (1, 14),
    "NLPTwitterAnalysisClustering": (1, 16),
    "HamshahriClustring": (1, 170),
    "SIDClustring": (1, 15),
    "MIRACLReranking": (3, 5),
    "WikipediaRerankingMultilingual": (3, 5),
    "SAMSumFa": (3, 12),
    "SynPerChatbotSumSRetrieval": (3, 1),
    "SynPerChatbotRAGSumSRetrieval": (3, 1),
    "SynPerQARetrieval": (3, 5),
    "SynPerChatbotTopicsRetrieval": (3, 14),
    "SynPerChatbotRAGTopicsRetrieval": (3, 14),
    "SynPerChatbotRAGFAQRetrieval": (3, 3),
    "PersianWebDocumentRetrieval": (3, 13),
}

# Add all retrieval datasets with task (3, 13)
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
]

for dataset in RETRIEVAL_DATASETS:
    DATASET_TASKS[dataset] = (3, 13)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class APIError(Exception):
    """Custom exception for API-related errors."""

    def __init__(
        self, message: str, status_code: int = None, response_data: dict = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class TaskProcessor:
    """Handles task-specific text preprocessing."""

    @staticmethod
    def preprocess_sample(
        sample: str | list[str],
        dataset_name: str,
        prompt_type: PromptType | None,
        sub: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """Preprocess sample based on task type."""
        # Skip preprocessing for hakim_unsuper model
        if model_name and "unsupe" in model_name:
            logging.info(f"Skipping preprocessing for unsupervised model: {model_name}")
            return str(sample)

        task_id, subtask_id = DATASET_TASKS.get(dataset_name, (None, None))

        if task_id is None:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return str(sample)

        # Skip processing for pair classification tasks
        if task_id == 4:
            return str(sample)

        task_key = f"{task_id}_{subtask_id}"
        task_prompt, _ = TASK_CONFIGS.get(task_key, ("", None))

        if not task_prompt:
            return str(sample)

        task_prompt = f"مسئله : {task_prompt}"

        # Single text classification
        if task_id == 1:
            return f"{task_prompt} | متن : {sample}"

        # Retrieval tasks
        elif task_id == 3:
            if sub == "sentence1":
                return f"{task_prompt} | متن اول : {sample}"
            elif sub == "sentence2":
                return f"{task_prompt} | متن دوم : {sample}"
            elif prompt_type and prompt_type.value == "query":
                return f"{task_prompt} | متن اول : {sample}"
            elif prompt_type and prompt_type.value == "passage":
                return f"{task_prompt} | متن دوم : {sample}"

        return str(sample)


class OurInstructModelWrapper(Wrapper):
    """Wrapper for the Hakim instruction-following model."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        max_retries: int = 3,
        retry_delay: int = 10,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_url = f"https://mcinext.ai/api/{model_name}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Get API key from environment
        self.api_key = self._get_api_key()

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.processor = TaskProcessor()
        logger.info(f"Initialized model wrapper for: {model_name}")

    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv("MCINEXT_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Please set MCINEXT_API_KEY environment variable."
            )

        return api_key

    def to(self, device: torch.device) -> None:
        """Move model to device (no-op for API-based model)."""
        pass

    def _validate_embedding_item(self, item: Any, index: int) -> list[str]:
        """Validate individual embedding item and return list of errors."""
        errors = []

        if not isinstance(item, dict):
            errors.append(
                f"Item {index}: Expected dictionary, got {type(item).__name__}"
            )
            return errors

        if "embedding" not in item:
            errors.append(
                f"Item {index}: Missing 'embedding' field. Available fields: {list(item.keys())}"
            )
            return errors

        embedding = item["embedding"]
        if not isinstance(embedding, list):
            errors.append(
                f"Item {index}: Embedding must be a list, got {type(embedding).__name__}"
            )
        elif len(embedding) == 0:
            errors.append(f"Item {index}: Embedding list is empty")
        elif not all(isinstance(x, (int, float)) for x in embedding):
            errors.append(f"Item {index}: Embedding contains non-numeric values")

        return errors

    def _validate_api_response(self, response_data: dict[str, Any]) -> None:
        """Validate API response structure with detailed error messages."""
        validation_errors = []

        if not isinstance(response_data, dict):
            raise ValidationError(
                "API response must be a dictionary",
                {"received_type": type(response_data).__name__},
            )

        if "data" not in response_data:
            raise ValidationError(
                "API response missing required 'data' field",
                {"available_fields": list(response_data.keys())},
            )

        data = response_data["data"]
        if not isinstance(data, list):
            raise ValidationError(
                "API response 'data' field must be a list",
                {"received_type": type(data).__name__},
            )

        for i, item in enumerate(data):
            item_errors = self._validate_embedding_item(item, i)
            validation_errors.extend(item_errors)

        if validation_errors:
            raise ValidationError(
                f"API response validation failed with {len(validation_errors)} errors",
                {"errors": validation_errors},
            )

    def _make_api_request(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make API request with retry logic and comprehensive error handling."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"API request attempt {attempt + 1}/{self.max_retries} for {len(data.get('input', []))} items"
                )

                response = requests.post(
                    self.api_url, headers=self.headers, json=data, timeout=60
                )
                response.raise_for_status()

                response_data = response.json()
                self._validate_api_response(response_data)

                logger.info(f"API request successful on attempt {attempt + 1}")
                return response_data

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timeout on attempt {attempt + 1}: {str(e)}")

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = response.status_code if "response" in locals() else None

                if status_code == 429:  # Rate limiting
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited (429) on attempt {attempt + 1}, waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                elif status_code and status_code >= 500:  # Server errors
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Server error {status_code} on attempt {attempt + 1}, waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    # Client errors (4xx except 429) - don't retry
                    error_msg = f"API client error {status_code}: {str(e)}"
                    logger.error(error_msg)
                    raise APIError(
                        error_msg, status_code, getattr(response, "json", lambda: {})()
                    )

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                wait_time = self.retry_delay * (attempt + 1)
                logger.warning(
                    f"Connection error on attempt {attempt + 1}: {str(e)}, waiting {wait_time}s"
                )
                time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                last_exception = e
                wait_time = self.retry_delay * (attempt + 1)
                logger.warning(
                    f"Request error on attempt {attempt + 1}: {str(e)}, waiting {wait_time}s"
                )
                time.sleep(wait_time)

            except ValidationError as e:
                last_exception = e
                logger.error(
                    f"Response validation failed on attempt {attempt + 1}: {str(e)}"
                )
                if e.details:
                    logger.error(f"Validation details: {e.details}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except ValueError as e:
                last_exception = e
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # All retries failed
        error_msg = f"API request failed after {self.max_retries} attempts"
        logger.error(f"{error_msg}. Last error: {str(last_exception)}")
        raise APIError(
            error_msg,
            details={
                "last_exception": str(last_exception),
                "attempts": self.max_retries,
            },
        )

    def _validate_inputs(self, sentences: list[str], batch_size: int) -> None:
        """Validate input parameters."""
        if not sentences:
            raise ValueError("Input sentences list is empty")

        if batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size: {batch_size}. Must be a positive integer."
            )

        # Check for empty or None sentences
        for i, sentence in enumerate(sentences):
            if sentence is None:
                raise ValueError(f"Sentence at index {i} is None")
            if not isinstance(sentence, str):
                raise ValueError(
                    f"Sentence at index {i} is not a string: {type(sentence)}"
                )

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode sentences using the API with batch processing and robust error handling."""
        # Validate inputs
        self._validate_inputs(sentences, batch_size)

        sub = kwargs.get("sub", None)

        # Preprocess sentences
        try:
            processed_sentences = [
                self.processor.preprocess_sample(
                    sentence, task_name, prompt_type, sub, self.model_name
                )
                for sentence in sentences
            ]
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise ValueError(f"Failed to preprocess sentences: {e}") from e

        logger.debug(
            f"Processing {len(processed_sentences)} sentences for task: {task_name}"
        )

        all_embeddings = []
        total_batches = (len(processed_sentences) + batch_size - 1) // batch_size

        # Process in batches
        for i in range(0, len(processed_sentences), batch_size):
            batch_num = i // batch_size + 1
            batch = processed_sentences[i : i + batch_size]

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} with {len(batch)} sentences"
            )

            # Prepare API request for current batch
            data = {
                "model": self.model_name,
                "input": batch,
                "encoding_format": "float",
                "add_special_tokens": True,
            }

            # Make API call for current batch with retry logic
            try:
                result = self._make_api_request(data)

                # Extract embeddings with proper error handling
                batch_embeddings = []
                for item in result["data"]:
                    embedding = item["embedding"]
                    if not embedding:  # Check for empty embeddings
                        raise ValidationError("Received empty embedding from API")
                    batch_embeddings.append(embedding)

                # Verify we got the expected number of embeddings
                if len(batch_embeddings) != len(batch):
                    raise ValidationError(
                        f"Expected {len(batch)} embeddings, got {len(batch_embeddings)}"
                    )

                all_embeddings.extend(batch_embeddings)

            except (APIError, ValidationError) as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error processing batch {batch_num}: {e}")
                raise APIError(
                    f"Unexpected error processing batch {batch_num}: {e}"
                ) from e

        # Verify final results
        if len(all_embeddings) != len(sentences):
            raise ValidationError(
                f"Final embedding count mismatch: expected {len(sentences)}, got {len(all_embeddings)}"
            )

        try:
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            logger.debug(f"Generated embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
        except Exception as e:
            logger.error(f"Failed to convert embeddings to numpy array: {e}")
            raise ValueError(f"Failed to convert embeddings to numpy array: {e}") from e


# Model metadata
hakim = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim",
        revision="v1",
    ),
    name="MCINext/Hakim",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail": [],
        "SAMSumFa": ["train"],
        "SynPerChatbotSumSRetrieval": ["train"],
        "SynPerChatbotRAGSumSRetrieval": ["train"],
        "SynPerChatbotConvSAClassification": ["train"],
        "SynPerChatbotConvSAToneChatbotClassification": ["train"],
        "SynPerChatbotConvSAToneUserClassification": ["train"],
        "SynPerChatbotSatisfactionLevelClassification": ["train"],
        "SynPerChatbotRAGToneChatbotClassification": ["train"],
        "SynPerChatbotRAGToneUserClassification": ["train"],
        "SynPerChatbotToneChatbotClassification": ["train"],
        "SynPerChatbotToneUserClassification": ["train"],
        "SynPerTextToneClassification": ["train"],
        "SIDClassification": ["train"],
        "PersianTextEmotion": ["train"],
        "SentimentDKSF": ["train"],
        "NLPTwitterAnalysisClassification": ["train"],
        "DigikalamagClassification": ["train"],
        "DigikalamagClustering": ["train"],
        "NLPTwitterAnalysisClustering": ["train"],
        "SIDClustring": ["train"],
        "CExaPPC": ["train"],
        "SynPerChatbotRAGFAQPC": ["train"],
        "FarsiParaphraseDetection": ["train"],
        "SynPerTextKeywordsPC": ["train"],
        "SynPerQAPC": ["train"],
        "ParsinluEntail": ["train"],
        "ParsinluQueryParaphPC": ["train"],
        "FiQA2018-Fa": ["train"],
        "HotpotQA-Fa": ["train"],
        "MSMARCO-Fa": ["train"],
        "NFCorpus-Fa": ["train"],
        "SciFact-Fa": ["train"],
        "SynPerQARetrieval": ["train"],
        "SynPerChatbotTopicsRetrieval": ["train"],
        "SynPerChatbotRAGTopicsRetrieval": ["train"],
        "SynPerChatbotRAGFAQRetrieval": ["train"],
        "Farsick": ["train"],
        "SynPerSTS": ["train"],
        "Query2Query": ["train"],
    },
)


hakim_small = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim-small",
        revision="v1",
    ),
    name="MCINext/Hakim-small",
    languages=["fas-Arab"],
    open_weights=False,
    revision="1",
    release_date="2025-05-10",
    n_parameters=38_736_384,
    memory_usage_mb=148,
    embed_dim=512,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/MCINext/Hakim-small",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail": [],
        "SAMSumFa": ["train"],
        "SynPerChatbotSumSRetrieval": ["train"],
        "SynPerChatbotRAGSumSRetrieval": ["train"],
        "SynPerChatbotConvSAClassification": ["train"],
        "SynPerChatbotConvSAToneChatbotClassification": ["train"],
        "SynPerChatbotConvSAToneUserClassification": ["train"],
        "SynPerChatbotSatisfactionLevelClassification": ["train"],
        "SynPerChatbotRAGToneChatbotClassification": ["train"],
        "SynPerChatbotRAGToneUserClassification": ["train"],
        "SynPerChatbotToneChatbotClassification": ["train"],
        "SynPerChatbotToneUserClassification": ["train"],
        "SynPerTextToneClassification": ["train"],
        "SIDClassification": ["train"],
        "PersianTextEmotion": ["train"],
        "SentimentDKSF": ["train"],
        "NLPTwitterAnalysisClassification": ["train"],
        "DigikalamagClassification": ["train"],
        "DigikalamagClustering": ["train"],
        "NLPTwitterAnalysisClustering": ["train"],
        "SIDClustring": ["train"],
        "CExaPPC": ["train"],
        "SynPerChatbotRAGFAQPC": ["train"],
        "FarsiParaphraseDetection": ["train"],
        "SynPerTextKeywordsPC": ["train"],
        "SynPerQAPC": ["train"],
        "ParsinluEntail": ["train"],
        "ParsinluQueryParaphPC": ["train"],
        "FiQA2018-Fa": ["train"],
        "HotpotQA-Fa": ["train"],
        "MSMARCO-Fa": ["train"],
        "NFCorpus-Fa": ["train"],
        "SciFact-Fa": ["train"],
        "SynPerQARetrieval": ["train"],
        "SynPerChatbotTopicsRetrieval": ["train"],
        "SynPerChatbotRAGTopicsRetrieval": ["train"],
        "SynPerChatbotRAGFAQRetrieval": ["train"],
        "Farsick": ["train"],
        "SynPerSTS": ["train"],
        "Query2Query": ["train"],
    },
)

hakim_unsup = ModelMeta(
    loader=partial(
        OurInstructModelWrapper,
        trust_remote_code=True,
        model_name="hakim-unsup",
        revision="v1",
    ),
    name="MCINext/Hakim-unsup",
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
    training_datasets={
        "FarsTail": [],
        "Farsick": ["train"],
        "MSMARCO-Fa": ["train"],
        "Query2Query": ["train"],
    },
)
