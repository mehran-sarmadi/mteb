from __future__ import annotations

import logging
from functools import partial
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import tqdm
from transformers import AutoTokenizer
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


from .our_instructions import preprocess_sample
from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class OurInstructModelWrapper(Wrapper):

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):  
        print(f"\n\nModel to test is = {model_name}\n\n")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):


        # default to search_document if input_type and prompt_name are not provided
        print('\n\nyes you right...\n\n')
        print(task_name)
        # print(type(prompt_type.value))
        print(len(sentences))
        print('before...')
        print(sentences[0])
        sentences = [preprocess_sample(sentence, task_name, prompt_type, self.model_name) for sentence in sentences]
        # print(task_name)
        print('after...')
        print(sentences[0])

        emb = self.model.encode(sentences, batch_size=batch_size, **kwargs)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
        # emb = F.normalize(emb, p=2, dim=1)
        if kwargs.get("convert_to_tensor", False):
            emb = emb.cpu().detach().numpy()
        return emb


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retrieval_2_9neg_instruct_stage3_v2",
#         revision="v1",
#     ),
#     name = "retrieval_2_9neg_instruct_stage3_v2",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )

# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retrieval_2_9neg_instruct_stage3_with_inbatch_v2",
#         revision="v1",
#     ),
#     name = "retrieval_2_9neg_instruct_stage3_with_inbatch_v2",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )

# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_v2",
#         revision="v1",
#     ),
#     name = "retro_9neg_instruct_stage3_v2",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_with_inbatch_v2",
#         revision="v1",
#     ),
#     name = "retro_9neg_instruct_stage3_with_inbatch_v2",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )

# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_9neg_instruct_stage3_v2_v4_no_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_9neg_instruct_stage3_v2_v4_no_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )



# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_v2_v4_no_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_v2_v4_no_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_v2_v4_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/retro_9neg_instruct_stage3_v2_v4_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )


out_instruct_model = ModelMeta(
    loader=partial(  
        OurInstructModelWrapper,
        model_name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_9neg_instruct_stage3_v2_v4_with_inbatch",
        revision="v1",
    ),
    name = "/mnt/data/ez-workspace/FlagEmbedding/FlagEmbedding/baai_general_embedding/results/bge_9neg_instruct_stage3_v2_v4_with_inbatch",
    languages=["fas_Arab"],
    revision="v1",
    release_date="2024-02-10",
    n_parameters=int(149 * 1e6),
    memory_usage_mb=284,
    embed_dim=768,
    license="apache-2",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-modernbert-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,  # couldn't find
    public_training_data=None,
    training_datasets={},  # English part of gte_multi_training_data,
    open_weights=True
)


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v3_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v3_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v5_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v5_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )


# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v6_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v6_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )

# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/bge_unbalanced_9neg_instruct_stage3_v2_v4_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/bge_unbalanced_9neg_instruct_stage3_v2_v4_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )

# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/bge_balanced_9neg_instruct_stage3_v2_v4_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/bge_balanced_9neg_instruct_stage3_v2_v4_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )



# out_instruct_model = ModelMeta(
#     loader=partial(  
#         OurInstructModelWrapper,
#         trust_remote_code=True,
#         model_name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v2_with_inbatch",
#         revision="v1",
#     ),
#     name = "/mnt/data/mehran-workspace/text-embedding/mteb-test/models_to_test/retro_9neg_instruct_stage3_v2_v2_with_inbatch",
#     languages=["fas_Arab"],
#     revision="v1",
#     release_date="2024-02-10",
# )