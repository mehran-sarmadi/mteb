from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ARCChallenge(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ARCChallenge",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.",
        reference="https://allenai.org/data/arc",
        dataset={
            "path": "RAR-b/ARC-Challenge",
            "revision": "c481e0da3dcbbad8bce7721dea9085b74320a0a3",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{clark2018think,
  title={Think you have solved question answering? try arc, the ai2 reasoning challenge},
  author={Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal={arXiv preprint arXiv:1803.05457},
  year={2018}
}
""",
        prompt={"query": "Retrieve the answer to the question."},
    )
