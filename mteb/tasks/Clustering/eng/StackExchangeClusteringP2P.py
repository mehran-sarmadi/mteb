from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering
from ....abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)


class StackExchangeClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2P.v2",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering-p2p",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{geigle:2021:arxiv,
  archiveprefix = {arXiv},
  author = {Gregor Geigle and
Nils Reimers and
Andreas R{\"u}ckl{\'e} and
Iryna Gurevych},
  eprint = {2104.07081},
  journal = {arXiv preprint},
  title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  url = {http://arxiv.org/abs/2104.07081},
  volume = {abs/2104.07081},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of StackExchange posts based on the given paragraphs",
        adapted_from=["StackExchangeClusteringP2P"],
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            # Remove sentences and labels with only 1 label example.
            unique_labels, counts = np.unique(labels, return_counts=True)
            solo_label_idx = np.where(counts == 1)
            solo_labels = unique_labels[solo_label_idx]
            for solo_label in solo_labels:
                loc = labels.index(solo_label)
                labels.pop(loc)
                sentences.pop(loc)
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})

        self.dataset = DatasetDict(ds)


class StackExchangeClusteringP2P(AbsTaskClustering):
    superseded_by = "StackExchangeClusteringP2P.v2"
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2P",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering-p2p",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{geigle:2021:arxiv,
  archiveprefix = {arXiv},
  author = {Gregor Geigle and
Nils Reimers and
Andreas R{\"u}ckl{\'e} and
Iryna Gurevych},
  eprint = {2104.07081},
  journal = {arXiv preprint},
  title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  url = {http://arxiv.org/abs/2104.07081},
  volume = {abs/2104.07081},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of StackExchange posts based on the given paragraphs",
    )
