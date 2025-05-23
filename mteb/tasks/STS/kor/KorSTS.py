from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class KorSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorSTS",
        dataset={
            "path": "dkoterwa/kor-sts",
            "revision": "016f35f9b961daaaa7a352e927084e3da662ac1f",
        },
        description="Benchmark dataset for STS in Korean. Created by machine translation and human post editing of the STS-B dataset.",
        reference="https://arxiv.org/abs/2004.03289",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2017-01-01"),  # rough approximates
        domains=["News", "Web"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated and localized",
        bibtex_citation=r"""
@article{ham2020kornli,
  author = {Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal = {arXiv preprint arXiv:2004.03289},
  title = {KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  year = {2020},
}
""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
