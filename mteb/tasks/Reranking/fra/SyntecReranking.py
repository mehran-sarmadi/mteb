from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SyntecReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SyntecReranking",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "revision": "daf0863838cd9e3ba50544cdce3ac2b338a1b0ad",
        },
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2022-12-01", "2022-12-02"),
        domains=["Legal", "Written"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{ciancone2024extending,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {Extending the Massive Text Embedding Benchmark to French},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
            split=self.metadata.eval_splits[0],
        )
        documents = datasets.load_dataset(
            name="documents", **self.metadata_dict["dataset"], split="test"
        )
        # replace documents ids in positive and negative column by their respective texts
        doc_id2txt = dict(list(zip(documents["doc_id"], documents["text"])))

        self.dataset = self.dataset.map(
            lambda x: {
                "positive": [doc_id2txt[docid] for docid in x["positive"]],
                "negative": [doc_id2txt[docid] for docid in x["negative"]],
            }
        )
        self.dataset = datasets.DatasetDict({"test": self.dataset})

        self.dataset_transform()

        self.data_loaded = True
