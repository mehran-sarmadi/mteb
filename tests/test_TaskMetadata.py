from __future__ import annotations

import logging

import pytest

from mteb import AbsTask
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.overview import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
_HISTORIC_DATASETS = [
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC",
    "TNews",
    "IFlyTek",
    "MultilingualSentiment",
    "JDReview",
    "OnlineShopping",
    "Waimai",
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BigPatentClustering",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
    "WikiCitiesClustering",
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "EightTagsClustering",
    "RomaniBibleClustering",
    "SpanishNewsClusteringP2P",
    "SwednClustering",
    "CLSClusteringS2S",
    "CLSClusteringP2P",
    "ThuNewsClusteringS2S",
    "ThuNewsClusteringP2P",
    "TV2Nordretrieval",
    "TwitterHjerneRetrieval",
    "GerDaLIR",
    "GerDaLIRSmall",
    "GermanDPR",
    "GermanQuAD-Retrieval",
    "LegalQuAD",
    "AILACasedocs",
    "AILAStatutes",
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HagridRetrieval",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "MSMARCO",
    "MSMARCOv2",
    "NarrativeQARetrieval",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "SyntecRetrieval",
    "JaQuADRetrieval",
    "Ko-miracl",
    "Ko-StrategyQA",
    "MintakaRetrieval",
    "MIRACLRetrieval",
    "MultiLongDocRetrieval",
    "XMarket",
    "SNLRetrieval",
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
    "SpanishPassageRetrievalS2P",
    "SpanishPassageRetrievalS2S",
    "SweFaqRetrieval",
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
    "LeCaRDv2",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "OpusparcusPC",
    "PawsX",
    "SICK-E-PL",
    "PpcPC",
    "CDSC-E",
    "PSC",
    "Ocnli",
    "Cmnli",
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "AlloprofReranking",
    "SyntecReranking",
    "T2Reranking",
    "MMarcoReranking",
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "GermanSTSBenchmark",
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "FinParaSTS",
    "SICKFr",
    "KLUE-STS",
    "KorSTS",
    "STS17",
    "STS22",
    "STSBenchmarkMultilingualSTS",
    "SICK-R-PL",
    "CDSC-R",
    "RonSTS",
    "STSES",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
    "SummEval",
    "SummEvalFr",
    "MalayalamNewsClassification",
    "TamilNewsClassification",
    "TenKGnadClusteringP2P.v2",
    "TenKGnadClusteringS2S.v2",
    "ClimateFEVER-Fa",
    "DBPedia-Fa",
    "HotpotQA-Fa",
    "MSMARCO-Fa",
    "NQ-Fa",
    "ArguAna-Fa",
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
    "FiQA2018-Fa",
    "NFCorpus-Fa",
    "QuoraRetrieval-Fa",
    "SCIDOCS-Fa",
    "SciFact-Fa",
    "TRECCOVID-Fa",
    "Touche2020-Fa",
]


def test_given_dataset_config_then_it_is_valid():
    my_task = TaskMetadata(
        name="MyTask",
        dataset={
            "path": "test/dataset",
            "revision": "1.0",
        },
        description="testing",
        reference=None,
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=None,
        domains=None,
        license=None,
        task_subtypes=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
    )
    assert my_task.dataset["path"] == "test/dataset"
    assert my_task.dataset["revision"] == "1.0"


def test_given_missing_dataset_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(  # type: ignore
            name="MyTask",
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_missing_revision_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_none_revision_path_then_it_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        my_task = TaskMetadata(
            name="MyTask",
            dataset={"path": "test/dataset", "revision": None},
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )

        assert my_task.dataset["revision"] is None

        warning_logs = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warning_logs) == 1
        assert (
            warning_logs[0].message
            == "Revision missing for the dataset test/dataset. "
            + "It is encourage to specify a dataset revision for reproducability."
        )


def test_unfilled_metadata_is_not_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        ).is_filled()
        is False
    )


def test_filled_metadata_is_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference="https://aclanthology.org/W19-6138/",
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=("2021-01-01", "2021-12-31"),
            domains=["Non-fiction", "Written"],
            license="mit",
            task_subtypes=["Thematic clustering"],
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
            bibtex_citation="Someone et al",
        ).is_filled()
        is True
    )


def test_all_metadata_is_filled_and_valid():
    all_tasks = get_tasks()

    unfilled_metadata = []
    for task in all_tasks:
        if (
            task.metadata.name not in _HISTORIC_DATASETS
            and task.metadata.name.replace("HardNegatives", "")
            not in _HISTORIC_DATASETS
        ):
            if not task.metadata.is_filled() and (
                not task.metadata.validate_metadata()
            ):
                unfilled_metadata.append(task.metadata.name)
    if unfilled_metadata:
        raise ValueError(
            f"The metadata of the following datasets is not filled: {unfilled_metadata}"
        )


def test_disallow_trust_remote_code_in_new_datasets():
    # DON'T ADD NEW DATASETS TO THIS LIST
    # THIS IS ONLY INTENDED FOR HISTORIC DATASETS
    exceptions = [
        "BornholmBitextMining",
        "BibleNLPBitextMining",
        "DiaBlaBitextMining",
        "FloresBitextMining",
        "IN22ConvBitextMining",
        "IN22GenBitextMining",
        "IndicGenBenchFloresBitextMining",
        "IWSLT2017BitextMining",
        "NTREXBitextMining",
        "SRNCorpusBitextMining",
        "VieMedEVBitextMining",
        "HotelReviewSentimentClassification",
        "TweetEmotionClassification",
        "DanishPoliticalCommentsClassification",
        "TenKGnadClassification",
        "ArxivClassification",
        "FinancialPhrasebankClassification",
        "FrenkEnClassification",
        "PatentClassification",
        "PoemSentimentClassification",
        "TweetTopicSingleClassification",
        "YahooAnswersTopicsClassification",
        "FilipinoHateSpeechClassification",
        "HebrewSentimentAnalysis",
        "HindiDiscourseClassification",
        "FrenkHrClassification",
        "Itacola",
        "JavaneseIMDBClassification",
        "WRIMEClassification",
        "KorHateClassification",
        "KorSarcasmClassification",
        "AfriSentiClassification",
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "NaijaSenti",
        "NordicLangClassification",
        "NusaX-senti",
        "SwissJudgementClassification",
        "MyanmarNews",
        "DutchBookReviewSentimentClassification",
        "NorwegianParliamentClassification",
        "PAC",
        "HateSpeechPortugueseClassification",
        "Moroco",
        "RomanianReviewsSentiment",
        "RomanianSentimentClassification",
        "GeoreviewClassification",
        "FrenkSlClassification",
        "DalajClassification",
        "SwedishSentimentClassification",
        "WisesightSentimentClassification",
        "UrduRomanSentimentClassification",
        "VieStudentFeedbackClassification",
        "IndicReviewsClusteringP2P",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "MLSUMClusteringP2P.v2",
        "CodeSearchNetRetrieval",
        "DanFEVER",
        "GerDaLIR",
        "GermanDPR",
        "AlphaNLI",
        "ARCChallenge",
        "FaithDial",
        "HagridRetrieval",
        "HellaSwag",
        "PIQA",
        "Quail",
        "RARbCode",
        "RARbMath",
        "SIQA",
        "SpartQA",
        "TempReasonL1",
        "TempReasonL2Context",
        "TempReasonL2Fact",
        "TempReasonL2Pure",
        "TempReasonL3Context",
        "TempReasonL3Fact",
        "TempReasonL3Pure",
        "TopiOCQA",
        "WinoGrande",
        "AlloprofRetrieval",
        "BSARDRetrieval",
        "JaGovFaqsRetrieval",
        "JaQuADRetrieval",
        "NLPJournalAbsIntroRetrieval",
        "NLPJournalTitleAbsRetrieval",
        "NLPJournalTitleIntroRetrieval",
        "IndicQARetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "MLQARetrieval",
        "MultiLongDocRetrieval",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2023Retrieval",
        "XMarket",
        "XPQARetrieval",
        "ArguAna-PL",
        "DBPedia-PL",
        "FiQA-PL",
        "HotpotQA-PL",
        "MSMARCO-PL",
        "NFCorpus-PL",
        "NQ-PL",
        "Quora-PL",
        "SCIDOCS-PL",
        "SciFact-PL",
        "TRECCOVID-PL",
        "SpanishPassageRetrievalS2P",
        "SpanishPassageRetrievalS2S",
        "SwednRetrieval",
        "SweFaqRetrieval",
        "KorHateSpeechMLClassification",
        "BrazilianToxicTweetsClassification",
        "CTKFactsNLI",
        "LegalBenchPC",
        "indonli",
        "OpusparcusPC",
        "PawsX",
        "XStance",
        "MIRACLReranking",
        "FinParaSTS",
        "JSICK",
        "JSTS",
        "RonSTS",
        "STSES",
        "AlloProfClusteringP2P.v2",
        "AlloProfClusteringS2S.v2",
        "LivedoorNewsClustering",
        "MewsC16JaClustering",
        "MLSUMClusteringS2S.v2",
        "SwednClusteringP2P",
        "SwednClusteringS2S",
    ]

    assert (
        135 == len(exceptions)
    ), "The number of exceptions has changed. Please do not add new datasets to this list."

    exceptions = []

    for task in get_tasks():
        if task.metadata.dataset.get("trust_remote_code", False):
            assert (
                task.metadata.name not in exceptions
            ), f"Dataset {task.metadata.name} should not trust remote code"


def test_empy_descriptive_stat_in_new_datasets():
    # DON'T ADD NEW DATASETS TO THIS LIST
    # THIS IS ONLY INTENDED FOR HISTORIC DATASETS
    exceptions = [
        "TbilisiCityHallBitextMining",
        "BibleNLPBitextMining",
        "BUCC.v2",
        "DiaBlaBitextMining",
        "FloresBitextMining",
        "IN22GenBitextMining",
        "IndicGenBenchFloresBitextMining",
        "IWSLT2017BitextMining",
        "LinceMTBitextMining",
        "NollySentiBitextMining",
        "NorwegianCourtsBitextMining",
        "NTREXBitextMining",
        "NusaXBitextMining",
        "PhincBitextMining",
        "RomaTalesBitextMining",
        "Tatoeba",
        "SRNCorpusBitextMining",
        "VieMedEVBitextMining",
        "AJGT",
        "HotelReviewSentimentClassification",
        "OnlineStoreReviewSentimentClassification",
        "RestaurantReviewSentimentClassification",
        "TweetEmotionClassification",
        "TweetSarcasmClassification",
        "BengaliDocumentClassification",
        "BengaliHateSpeechClassification",
        "BengaliSentimentAnalysis",
        "BulgarianStoreReviewSentimentClassfication",
        "CSFDCZMovieReviewSentimentClassification",
        "CzechProductReviewSentimentClassification",
        "CzechSoMeSentimentClassification",
        "CzechSubjectivityClassification",
        "AngryTweetsClassification",
        "DanishPoliticalCommentsClassification",
        "DKHateClassification",
        "LccSentimentClassification",
        "GermanPoliticiansTwitterSentimentClassification",
        "TenKGnadClassification",
        "GreekLegalCodeClassification",
        "AmazonPolarityClassification",
        "ArxivClassification",
        "Banking77Classification",
        "DBpediaClassification",
        "EmotionClassification",
        "FinancialPhrasebankClassification",
        "FrenkEnClassification",
        "ImdbClassification",
        "CanadaTaxCourtOutcomesLegalBenchClassification",
        "ContractNLIConfidentialityOfAgreementLegalBenchClassification",
        "ContractNLIExplicitIdentificationLegalBenchClassification",
        "ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
        "ContractNLILimitedUseLegalBenchClassification",
        "ContractNLINoLicensingLegalBenchClassification",
        "ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
        "ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
        "ContractNLIPermissibleCopyLegalBenchClassification",
        "ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
        "ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
        "ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
        "ContractNLISharingWithEmployeesLegalBenchClassification",
        "ContractNLISharingWithThirdPartiesLegalBenchClassification",
        "ContractNLISurvivalOfObligationsLegalBenchClassification",
        "CorporateLobbyingLegalBenchClassification",
        "CUADAffiliateLicenseLicenseeLegalBenchClassification",
        "CUADAffiliateLicenseLicensorLegalBenchClassification",
        "CUADAntiAssignmentLegalBenchClassification",
        "CUADAuditRightsLegalBenchClassification",
        "CUADCapOnLiabilityLegalBenchClassification",
        "CUADChangeOfControlLegalBenchClassification",
        "CUADCompetitiveRestrictionExceptionLegalBenchClassification",
        "CUADCovenantNotToSueLegalBenchClassification",
        "CUADEffectiveDateLegalBenchClassification",
        "CUADExclusivityLegalBenchClassification",
        "CUADExpirationDateLegalBenchClassification",
        "CUADGoverningLawLegalBenchClassification",
        "CUADInsuranceLegalBenchClassification",
        "CUADIPOwnershipAssignmentLegalBenchClassification",
        "CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
        "CUADJointIPOwnershipLegalBenchClassification",
        "CUADLicenseGrantLegalBenchClassification",
        "CUADLiquidatedDamagesLegalBenchClassification",
        "CUADMinimumCommitmentLegalBenchClassification",
        "CUADMostFavoredNationLegalBenchClassification",
        "CUADNoSolicitOfCustomersLegalBenchClassification",
        "CUADNoSolicitOfEmployeesLegalBenchClassification",
        "CUADNonCompeteLegalBenchClassification",
        "CUADNonDisparagementLegalBenchClassification",
        "CUADNonTransferableLicenseLegalBenchClassification",
        "CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
        "CUADPostTerminationServicesLegalBenchClassification",
        "CUADPriceRestrictionsLegalBenchClassification",
        "CUADRenewalTermLegalBenchClassification",
        "CUADRevenueProfitSharingLegalBenchClassification",
        "CUADRofrRofoRofnLegalBenchClassification",
        "CUADSourceCodeEscrowLegalBenchClassification",
        "CUADTerminationForConvenienceLegalBenchClassification",
        "CUADThirdPartyBeneficiaryLegalBenchClassification",
        "CUADUncappedLiabilityLegalBenchClassification",
        "CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
        "CUADVolumeRestrictionLegalBenchClassification",
        "CUADWarrantyDurationLegalBenchClassification",
        "DefinitionClassificationLegalBenchClassification",
        "Diversity1LegalBenchClassification",
        "Diversity2LegalBenchClassification",
        "Diversity3LegalBenchClassification",
        "Diversity4LegalBenchClassification",
        "Diversity5LegalBenchClassification",
        "Diversity6LegalBenchClassification",
        "FunctionOfDecisionSectionLegalBenchClassification",
        "InsurancePolicyInterpretationLegalBenchClassification",
        "InternationalCitizenshipQuestionsLegalBenchClassification",
        "JCrewBlockerLegalBenchClassification",
        "LearnedHandsBenefitsLegalBenchClassification",
        "LearnedHandsBusinessLegalBenchClassification",
        "LearnedHandsConsumerLegalBenchClassification",
        "LearnedHandsCourtsLegalBenchClassification",
        "LearnedHandsCrimeLegalBenchClassification",
        "LearnedHandsDivorceLegalBenchClassification",
        "LearnedHandsDomesticViolenceLegalBenchClassification",
        "LearnedHandsEducationLegalBenchClassification",
        "LearnedHandsEmploymentLegalBenchClassification",
        "LearnedHandsEstatesLegalBenchClassification",
        "LearnedHandsFamilyLegalBenchClassification",
        "LearnedHandsHealthLegalBenchClassification",
        "LearnedHandsHousingLegalBenchClassification",
        "LearnedHandsImmigrationLegalBenchClassification",
        "LearnedHandsTortsLegalBenchClassification",
        "LearnedHandsTrafficLegalBenchClassification",
        "LegalReasoningCausalityLegalBenchClassification",
        "MAUDLegalBenchClassification",
        "NYSJudicialEthicsLegalBenchClassification",
        "OPP115DataRetentionLegalBenchClassification",
        "OPP115DataSecurityLegalBenchClassification",
        "OPP115DoNotTrackLegalBenchClassification",
        "OPP115FirstPartyCollectionUseLegalBenchClassification",
        "OPP115InternationalAndSpecificAudiencesLegalBenchClassification",
        "OPP115PolicyChangeLegalBenchClassification",
        "OPP115ThirdPartySharingCollectionLegalBenchClassification",
        "OPP115UserAccessEditAndDeletionLegalBenchClassification",
        "OPP115UserChoiceControlLegalBenchClassification",
        "OralArgumentQuestionPurposeLegalBenchClassification",
        "OverrulingLegalBenchClassification",
        "PersonalJurisdictionLegalBenchClassification",
        "PROALegalBenchClassification",
        "SCDBPAccountabilityLegalBenchClassification",
        "SCDBPAuditsLegalBenchClassification",
        "SCDBPCertificationLegalBenchClassification",
        "SCDBPTrainingLegalBenchClassification",
        "SCDBPVerificationLegalBenchClassification",
        "SCDDAccountabilityLegalBenchClassification",
        "SCDDAuditsLegalBenchClassification",
        "SCDDCertificationLegalBenchClassification",
        "SCDDTrainingLegalBenchClassification",
        "SCDDVerificationLegalBenchClassification",
        "TelemarketingSalesRuleLegalBenchClassification",
        "TextualismToolDictionariesLegalBenchClassification",
        "TextualismToolPlainLegalBenchClassification",
        "UCCVCommonLawLegalBenchClassification",
        "UnfairTOSLegalBenchClassification",
        "NewsClassification",
        "PatentClassification",
        "PoemSentimentClassification",
        "ToxicChatClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
        "TweetTopicSingleClassification",
        "YahooAnswersTopicsClassification",
        "YelpReviewFullClassification",
        "EstonianValenceClassification",
        "PersianFoodSentimentClassification",
        "FilipinoHateSpeechClassification",
        "FilipinoShopeeReviewsClassification",
        "FinToxicityClassification",
        "FrenchBookReviews",
        "MovieReviewSentimentClassification",
        "GujaratiNewsClassification",
        "HebrewSentimentAnalysis",
        "HindiDiscourseClassification",
        "SentimentAnalysisHindi",
        "FrenkHrClassification",
        "IndonesianIdClickbaitClassification",
        "IndonesianMongabayConservationClassification",
        "ItaCaseholdClassification",
        "Itacola",
        "JavaneseIMDBClassification",
        "WRIMEClassification",
        "KannadaNewsClassification",
        "KLUE-TC",
        "KorFin",
        "KorHateClassification",
        "KorSarcasmClassification",
        "KurdishSentimentClassification",
        "MalayalamNewsClassification",
        "MarathiNewsClassification",
        "MacedonianTweetSentimentClassification",
        "AfriSentiClassification",
        "AfriSentiLangClassification",
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "CataloniaTweetClassification",
        "CyrillicTurkicLangClassification",
        "HinDialectClassification",
        "IndicLangClassification",
        "IndicNLPNewsClassification",
        "IndicSentimentClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "MultiHateClassification",
        "MultilingualSentimentClassification",
        "NaijaSenti",
        "NordicLangClassification",
        "NusaParagraphEmotionClassification",
        "NusaParagraphTopicClassification",
        "NusaX-senti",
        "ScalaClassification",
        "SIB200Classification",
        "SouthAfricanLangClassification",
        "SwissJudgementClassification",
        "TurkicClassification",
        "TweetSentimentClassification",
        "MyanmarNews",
        "NepaliNewsClassification",
        "DutchBookReviewSentimentClassification",
        "NoRecClassification",
        "NorwegianParliamentClassification",
        "OdiaNewsClassification",
        "PunjabiNewsClassification",
        "CBD",
        "PolEmo2.0-IN",
        "PolEmo2.0-OUT",
        "AllegroReviews",
        "PAC",
        "HateSpeechPortugueseClassification",
        "Moroco",
        "RomanianReviewsSentiment",
        "RomanianSentimentClassification",
        "GeoreviewClassification",
        "HeadlineClassification",
        "InappropriatenessClassification",
        "KinopoiskClassification",
        "RuReviewsClassification",
        "RuSciBenchGRNTIClassification",
        "RuSciBenchOECDClassification",
        "SanskritShlokasClassification",
        "SinhalaNewsClassification",
        "SinhalaNewsSourceClassification",
        "CSFDSKMovieReviewSentimentClassification",
        "FrenkSlClassification",
        "SpanishNewsClassification",
        "SpanishSentimentClassification",
        "SiswatiNewsClassification",
        "SlovakMovieReviewSentimentClassification",
        "SwahiliNewsClassification",
        "DalajClassification",
        "SwedishSentimentClassification",
        "SweRecClassification",
        "TamilNewsClassification",
        "TeluguAndhraJyotiNewsClassification",
        "WisesightSentimentClassification",
        "TswanaNewsClassification",
        "TurkishMovieSentimentClassification",
        "TurkishProductSentimentClassification",
        "UkrFormalityClassification",
        "UrduRomanSentimentClassification",
        "VieStudentFeedbackClassification",
        "TNews",
        "IFlyTek",
        "MultilingualSentiment",
        "JDReview",
        "OnlineShopping",
        "Waimai",
        "YueOpenriceReviewClassification",
        "IsiZuluNewsClassification",
        "WikiCitiesClustering",
        "IndicReviewsClusteringP2P",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "RomaniBibleClustering",
        "SpanishNewsClusteringP2P",
        "BlurbsClusteringP2P.v2",
        "BlurbsClusteringS2S.v2",
        "TenKGnadClusteringP2P.v2",
        "TenKGnadClusteringS2S.v2",
        "ArXivHierarchicalClusteringS2S",
        "BigPatentClustering.v2",
        "BiorxivClusteringP2P.v2",
        "BiorxivClusteringS2S.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "RedditClustering.v2",
        "RedditClusteringP2P.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
        "TwentyNewsgroupsClustering.v2",
        "AlloProfClusteringP2P.v2",
        "AlloProfClusteringS2S.v2",
        "HALClusteringS2S.v2",
        "LivedoorNewsClustering.v2",
        "MewsC16JaClustering",
        "MLSUMClusteringP2P.v2",
        "MLSUMClusteringS2S.v2",
        "SIB200ClusteringS2S",
        "WikiClusteringP2P.v2",
        "SNLHierarchicalClusteringP2P",
        "SNLHierarchicalClusteringS2S",
        "VGHierarchicalClusteringP2P",
        "VGHierarchicalClusteringS2S",
        "EightTagsClustering.v2",
        "PlscClusteringS2S.v2",
        "PlscClusteringP2P.v2",
        "GeoreviewClusteringP2P",
        "RuSciBenchOECDClusteringP2P",
        "SwednClusteringP2P",
        "SwednClusteringS2S",
        "CLSClusteringS2S.v2",
        "CLSClusteringP2P.v2",
        "ThuNewsClusteringS2S.v2",
        "ThuNewsClusteringP2P.v2",
        "SadeemQuestionRetrieval",
        "DanFeverRetrieval",
        "TV2Nordretrieval",
        "TwitterHjerneRetrieval",
        "GerDaLIR",
        "GerDaLIRSmall",
        "GermanDPR",
        "GermanGovServiceRetrieval",
        "GermanQuAD-Retrieval",
        "LegalQuAD",
        "GreekCivicsQA",
        "AILACasedocs",
        "AILAStatutes",
        "AlphaNLI",
        "ARCChallenge",
        "ArguAna",
        "BrightRetrieval",
        "ClimateFEVER",
        "ClimateFEVERHardNegatives",
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "DBPedia",
        "DBPediaHardNegatives",
        "FaithDial",
        "FeedbackQARetrieval",
        "FEVER",
        "FEVERHardNegatives",
        "FiQA2018",
        "HagridRetrieval",
        "HellaSwag",
        "HotpotQA",
        "HotpotQAHardNegatives",
        "LegalBenchConsumerContractsQA",
        "LegalBenchCorporateLobbying",
        "LegalSummarization",
        "LEMBNarrativeQARetrieval",
        "LEMBNeedleRetrieval",
        "LEMBPasskeyRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBSummScreenFDRetrieval",
        "LEMBWikimQARetrieval",
        "LitSearchRetrieval",
        "MedicalQARetrieval",
        "MLQuestions",
        "MSMARCO",
        "MSMARCOHardNegatives",
        "MSMARCOv2",
        "NarrativeQARetrieval",
        "NFCorpus",
        "NQ",
        "NQHardNegatives",
        "PIQA",
        "Quail",
        "QuoraRetrieval",
        "QuoraRetrievalHardNegatives",
        "RARbCode",
        "RARbMath",
        "SCIDOCS",
        "SciFact",
        "SIQA",
        "SpartQA",
        "TempReasonL1",
        "TempReasonL2Context",
        "TempReasonL2Fact",
        "TempReasonL2Pure",
        "TempReasonL3Context",
        "TempReasonL3Fact",
        "TempReasonL3Pure",
        "TopiOCQA",
        "TopiOCQAHardNegatives",
        "TRECCOVID",
        "WinoGrande",
        "EstQA",
        "AlloprofRetrieval",
        "BSARDRetrieval",
        "FQuADRetrieval",
        "SyntecRetrieval",
        "HunSum2AbstractiveRetrieval",
        "JaGovFaqsRetrieval",
        "JaQuADRetrieval",
        "NLPJournalAbsIntroRetrieval",
        "NLPJournalTitleAbsRetrieval",
        "NLPJournalTitleIntroRetrieval",
        "GeorgianFAQRetrieval",
        "Ko-StrategyQA",
        "CrossLingualSemanticDiscriminationWMT19",
        "CrossLingualSemanticDiscriminationWMT21",
        "IndicQARetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        "MLQARetrieval",
        "MrTidyRetrieval",
        "MultiLongDocRetrieval",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2022RetrievalHardNegatives",
        "NeuCLIR2023Retrieval",
        "NeuCLIR2023RetrievalHardNegatives",
        "PublicHealthQA",
        "StatcanDialogueDatasetRetrieval",
        "WikipediaRetrievalMultilingual",
        "XMarket",
        "XPQARetrieval",
        "XQuADRetrieval",
        "NorQuadRetrieval",
        "SNLRetrieval",
        "ArguAna-PL",
        "DBPedia-PL",
        "DBPedia-PLHardNegatives",
        "FiQA-PL",
        "HotpotQA-PL",
        "HotpotQA-PLHardNegatives",
        "MSMARCO-PL",
        "MSMARCO-PLHardNegatives",
        "NFCorpus-PL",
        "NQ-PL",
        "NQ-PLHardNegatives",
        "Quora-PL",
        "Quora-PLHardNegatives",
        "SCIDOCS-PL",
        "SciFact-PL",
        "TRECCOVID-PL",
        "RiaNewsRetrieval",
        "RiaNewsRetrievalHardNegatives",
        "RuBQRetrieval",
        "SKQuadRetrieval",
        "SlovakSumRetrieval",
        "SpanishPassageRetrievalS2P",
        "SpanishPassageRetrievalS2S",
        "SwednRetrieval",
        "SweFaqRetrieval",
        "TurHistQuadRetrieval",
        "VieQuADRetrieval",
        "T2Retrieval",
        "MMarcoRetrieval",
        "DuRetrieval",
        "CovidRetrieval",
        "CmedqaRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "VideoRetrieval",
        "LeCaRDv2",
        "News21InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "KorHateSpeechMLClassification",
        "MalteseNewsClassification",
        "BrazilianToxicTweetsClassification",
        "SensitiveTopicsClassification",
        "ArEntail",
        "CTKFactsNLI",
        "FalseFriendsGermanEnglish",
        "LegalBenchPC",
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "FarsTail",
        "ArmenianParaphrasePC",
        "indonli",
        "KLUE-NLI",
        "OpusparcusPC",
        "RTE3",
        "XNLIV2",
        "XStance",
        "SICK-E-PL",
        "PpcPC",
        "CDSC-E",
        "PSC",
        "Assin2RTE",
        "SICK-BR-PC",
        "TERRa",
        "Ocnli",
        "Cmnli",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
        "WebLINXCandidatesReranking",
        "AlloprofReranking",
        "SyntecReranking",
        "VoyageMMarcoReranking",
        "MIRACLReranking",
        "RuBQReranking",
        "T2Reranking",
        "MMarcoReranking",
        "CMedQAv1-reranking",
        "CMedQAv2-reranking",
        "CPUSpeedTask",
        "GPUSpeedTask",
        "GermanSTSBenchmark",
        "BIOSSES",
        "SICK-R",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "FaroeseSTS",
        "FinParaSTS",
        "SICKFr",
        "JSICK",
        "JSTS",
        "KLUE-STS",
        "KorSTS",
        "IndicCrosslingualSTS",
        "SemRel24STS",
        "STS22.v2",
        "STSBenchmarkMultilingualSTS",
        "SICK-R-PL",
        "CDSC-R",
        "Assin2STS",
        "SICK-BR-STS",
        "RonSTS",
        "RUParaPhraserSTS",
        "RuSTSBenchmarkSTS",
        "STSES",
        "ATEC",
        "BQ",
        "LCQMC",
        "PAWSX",
        "STSB",
        "AFQMC",
        "QBQTC",
        "SummEvalSummarization.v2",
        "SummEvalFrSummarization.v2",
    ]

    assert (
        553 == len(exceptions)
    ), "The number of exceptions has changed. Please do not add new datasets to this list."

    exceptions = []

    for task in get_tasks():
        if task.metadata.descriptive_stats is None:
            assert (
                task.metadata.name not in exceptions
            ), f"Dataset {task.metadata.name} should have descriptive stats"


@pytest.mark.parametrize("task", get_tasks())
def test_eval_langs_correctly_specified(task: AbsTask):
    if task.is_multilingual:
        assert isinstance(
            task.metadata.eval_langs, dict
        ), f"{task.metadata.name} should have eval_langs as a dict"
    else:
        assert isinstance(
            task.metadata.eval_langs, list
        ), f"{task.metadata.name} should have eval_langs as a list"
