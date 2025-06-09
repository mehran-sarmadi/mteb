from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import requests
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)

dataset_info_dict_2 = {
    ###################################################################
    ########################## Classfication ##########################
    "PersianTextEmotion": {
        "task_id": 1,
        "subtask_id": 18,
    },
    "PersianFoodSentimentClassification": {
        "task_id": 1,
        "subtask_id": 190,
    },
    "SentimentDKSF": {
        "task_id": 1,
        "subtask_id": 19,
    },
    "MassiveIntentClassification": {
        "task_id": 1,
        "subtask_id": 170,
    },
    "MassiveScenarioClassification": {
        "task_id": 1,
        "subtask_id": 17,
    },
    "SynPerChatbotConvSAAnger": {
        "task_id": 1,
        "subtask_id": 4,
    },
    "SynPerChatbotConvSASatisfaction": {
        "task_id": 1,
        "subtask_id": 5,
    },
    "SynPerChatbotConvSAFriendship": {
        "task_id": 1,
        "subtask_id": 6,
    },
    "SynPerChatbotConvSAFear": {
        "task_id": 1,
        "subtask_id": 7,
    },
    "SynPerChatbotConvSAJealousy": {
        "task_id": 1,
        "subtask_id": 8,
    },
    "SynPerChatbotConvSASurprise": {
        "task_id": 1,
        "subtask_id": 9,
    },
    "SynPerChatbotConvSALove": {
        "task_id": 1,
        "subtask_id": 10,
    },
    "SynPerChatbotConvSASadness": {
        "task_id": 1,
        "subtask_id": 11,
    },
    "SynPerChatbotConvSAHappiness": {
        "task_id": 1,
        "subtask_id": 12,
    },
    "SynPerChatbotConvSAToneChatbotClassification": {
        "task_id": 1,
        "subtask_id": 21,
    },
    "SynPerChatbotConvSAToneUserClassification": {
        "task_id": 1,
        "subtask_id": 20,
    },
    "PersianTextTone": {
        "task_id": 1,
        "subtask_id": 13,
    },
    "SynPerChatbotToneUserClassification": {
        "task_id": 1,
        "subtask_id": 2,
    },
    "SynPerChatbotToneChatbotClassification": {
        "task_id": 1,
        "subtask_id": 3,
    },
    "SynPerChatbotRAGToneUserClassification": {
        "task_id": 1,
        "subtask_id": 2,
    },
    "SynPerChatbotRAGToneChatbotClassification": {
        "task_id": 1,
        "subtask_id": 3,
    },
    "SynPerChatbotSatisfactionLevelClassification": {
        "task_id": 1,
        "subtask_id": 1,
    },
    "DigimagClassification": {
        "task_id": 1,
        "subtask_id": 14,
    },
    "NLPTwitterAnalysisClassification": {
        "task_id": 1,
        "subtask_id": 16,
    },
    "SIDClassification": {
        "task_id": 1,
        "subtask_id": 15,
    },
    "DeepSentiPers": {
        "task_id": 1,
        "subtask_id": 19,
    },
    "DigikalamagClassification": {
        "task_id": 1,
        "subtask_id": 14,
    },
    #######################################################################
    ########################## PairClassification ##########################
    "FarsTail": {
        "task_id": 4,
        "subtask_id": 6,
    },
    "ParsinluEntail": {
        "task_id": 4,
        "subtask_id": 6,
    },
    "ParsinluQueryParaphPC": {
        "task_id": 4,
        "subtask_id": 7,
    },
    "SynPerChatbotRAGFAQPC": {
        "task_id": 4,
        "subtask_id": 1,
    },
    "SynPerTextKeywordsPC": {
        "task_id": 4,
        "subtask_id": 2,
    },
    "SynPerQAPC": {
        "task_id": 4,
        "subtask_id": 3,
    },
    "CExaPPC": {
        "task_id": 4,
        "subtask_id": 7,
    },
    "FarsiParaphraseDetection": {
        "task_id": 4,
        "subtask_id": 7,
    },
    #######################################################################
    ########################## STS ########################################
    "Farsick": {
        "task_id": 3,
        "subtask_id": 6,
    },
    "Query2Query": {
        "task_id": 3,
        "subtask_id": 6,
    },
    "SynPerSTS": {
        "task_id": 3,
        "subtask_id": 6,
    },
    #######################################################################
    ########################## Clustring #################################
    "BeytooteClustering": {
        "task_id": 1,
        "subtask_id": 170,
    },
    "DigikalamagClustering": {
        "task_id": 1,
        "subtask_id": 14,
    },
    "NLPTwitterAnalysisClustering": {
        "task_id": 1,
        "subtask_id": 16,
    },
    "HamshahriClustring": {
        "task_id": 1,
        "subtask_id": 170,
    },
    "SIDClustring": {
        "task_id": 1,
        "subtask_id": 15,
    },
    #######################################################################
    ########################## Reranking #################################
    "MIRACLReranking": {
        "task_id": 3,
        "subtask_id": 5,
    },
    "WikipediaRerankingMultilingual": {
        "task_id": 3,
        "subtask_id": 5,
    },
    #######################################################################
    ########################## Retrieval #################################
    "ArguAna-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "ClimateFEVER-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackAndroidRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackEnglishRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackGamingRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackGisRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackMathematicaRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackPhysicsRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackProgrammersRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackStatsRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackTexRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackUnixRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackWebmastersRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "CQADupstackWordpressRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "DBPedia-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "FiQA2018-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "HotpotQA-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "MSMARCO-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "NFCorpus-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "NQ-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "QuoraRetrieval-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "SCIDOCS-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "SciFact-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "TRECCOVID-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "Touche2020-Fa": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "MIRACLRetrieval": {
        "task_id": 3,
        "subtask_id": 13,
    },
    "WikipediaRetrievalMultilingual": {
        "task_id": 3,
        "subtask_id": 13,
    },
    ##########################################################################
    "SynPerQARetrieval": {
        "task_id": 3,
        "subtask_id": 5,
    },
    "SynPerChatbotTopicsRetrieval": {
        "task_id": 3,
        "subtask_id": 14,
    },
    "SynPerChatbotRAGTopicsRetrieval": {
        "task_id": 3,
        "subtask_id": 14,
    },
    "SynPerChatbotRAGFAQRetrieval": {
        "task_id": 3,
        "subtask_id": 3,
    },
    "PersianWebDocumentRetrieval": {
        "task_id": 3,
        "subtask_id": 13,
    },
    #######################################################################
    ########################## Summary Retrieval #################################
    "SAMSumFa": {
        "task_id": 3,
        "subtask_id": 12,
    },
    "SynPerChatbotSumSRetrieval": {
        "task_id": 3,
        "subtask_id": 1,
    },
    "SynPerChatbotRAGSumSRetrieval": {
        "task_id": 3,
        "subtask_id": 1,
    },
}

task_prompt_dict_v3 = {
    "1_1": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        "task_classes": ["عالی", "خوب", "متوسط", "بد", "خیلی بد"],
    },
    "1_2": {
        "task_prompt": "مسئله : دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        "task_classes": ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    },
    "1_3": {
        "task_prompt": "مسئله : دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        "task_classes": ["رسمی", "عامیانه", "کودکانه", "لاتی", "عصبانی"],
    },
    "1_4": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس عصبانیت کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_5": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس رضایت کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_6": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس صمیمیت کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_7": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس ترس کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_8": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس حسادت کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_9": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس شگفتی کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_10": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس عشق کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_11": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس غصه کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_12": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس خوشحالی کاربر در مکالمه با چت بات",
        "task_classes": ["مثبت", "منفی"],
    },
    "1_13": {
        "task_prompt": "مسئله : دسته بندی , تشخیص لحن متن",
        "task_classes": ["عامیانه", "رسمی", "کودکانه", "ادبی"],
    },
    "1_14": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی موضوعی متن",
        "task_classes": [
            "بازی ویدیویی",
            "راهنمای خرید",
            "سلامت و زیبایی",
            "علم و تکنولوژی",
            "عمومی",
            "هنر و سینما",
            "کتاب و ادبیات",
        ],
    },
    "1_15": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی موضوعی متن",
        "task_classes": [
            "پزشکی",
            "کشاورزی و منابع طبیعی",
            "فنی مهندسی",
            "علوم پایه",
            "علوم انسانی",
            "هنر و معماری",
            "علمی تخصصی",
            "دامپزشکی",
        ],
    },
    "1_16": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی موضوعی متن",
        "task_classes": [
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
    },
    "1_17": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی موضوعی متن",
        "task_classes": [
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
    },
    "1_18": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی احساس متن",
        "task_classes": ["شادی", "غم", "خشم", "انزجار", "ترس", "تعجب"],
    },
    "1_19": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس رضایت متن",
        "task_classes": ["مثبت", "منفی", "خنثی"],
    },
    "1_20": {
        "task_prompt": "مسئله : دسته بندی , تشخیص لحن کاربر در مکالمه با چت بات",
        "task_classes": ["رسمی", "عامیانه", "کودکانه"],
    },
    "1_21": {
        "task_prompt": "مسئله : دسته بندی , تشخیص لحن چت بات در مکالمه ی کاربر با چت بات",
        "task_classes": ["رسمی", "عامیانه", "کودکانه"],
    },
    "2_1": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , متن اول مکالمه ی کاربر با چت بات به همراه پیام جدید کاربر است. آیا متن دوم که یک سوال و پاسخ است به متن اول مرتبط است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_2": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آیا متن دوم کلمات کلیدی متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_3": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آیا متن دوم پاسخ متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_4": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , نوع ارتباط معنایی متن دوم با متن اول چگونه است ؟",
        "task_classes": [
            "مفهوم کاملا یکسان",
            "مفهوم تقریبا یکسان",
            "مفهوم تا حدی یکسان",
            "مفهوم متفاوت در موضوع یکسان",
            "مفهوم کاملا متفاوت",
        ],
    },
    "2_5": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , نوع ارتباط معنایی متن دوم با متن اول چگونه است ؟",
        "task_classes": [
            "مفهوم کاملا یکسان",
            "مفهوم تقریبا یکسان",
            "مفهوم تا حدی یکسان",
            "مفهوم تا حدی متفاوت با جزئیات یکسان",
            "مفهوم متفاوت در موضوع یکسان",
            "مفهوم کاملا متفاوت",
        ],
    },
    "2_6": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , نوع ارتباط معنایی متن دوم با متن اول چگونه است ؟",
        "task_classes": ["مشابه", "متضاد", "بی ارتباط"],
    },
    "2_7": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آيا متن دوم بازنویسی متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_8": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , احساس رضایت از نظر متن دوم در متن اول چگونه است ؟",
        "task_classes": ["مثبت", "منفی", "خنثی"],
    },
    "2_9": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آيا متن دوم ترجمه ی فارسی متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_10": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آيا متن دوم ترجمه ی عربی متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_11": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , آيا متن دوم ترجمه ی انگلیسی متن اول است ؟",
        "task_classes": ["مثبت", "منفی"],
    },
    "2_12": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , متن دوم که یک موجودیت نامدار در متن اول است به کدام دسته تعلق دارد ؟",
        "task_classes": [
            "گروه‌های سیاسی",
            "اشخاص",
            "سازمان‌ها",
            "مکان‌ها",
            "رویدادها",
            "ملت‌ها",
        ],
    },
    "2_13": {
        "task_prompt": "مسئله : دسته بندی با دو ورودی , متن دوم که یک موجودیت نامدار در متن اول است به کدام دسته تعلق دارد ؟",
        "task_classes": ["اشخاص", "سازمان‌ها", "مکان‌ها"],
    },
    "3_1": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم خلاصه ی متن اول است ؟",
        "task_classes": None,
    },
    "3_2": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوعات استخراج شده ی متن اول است ؟",
        "task_classes": None,
    },
    "3_3": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات به همراه پیام جدید کاربر است. آیا متن دوم که یک سوال و پاسخ است به متن اول مرتبط است ؟",
        "task_classes": None,
    },
    "3_4": {
        "task_prompt": "مسئله : تشخیص ارتباط , آیا متن دوم کلمات کلیدی متن اول است ؟",
        "task_classes": None,
    },
    "3_5": {
        "task_prompt": "مسئله : تشخیص ارتباط , آیا متن دوم پاسخ متن اول است ؟",
        "task_classes": None,
    },
    "3_6": {
        "task_prompt": "مسئله : تشخیص ارتباط , آیا متن دوم شباهت معنایی با متن اول دارد ؟",
        "task_classes": None,
    },
    "3_7": {
        "task_prompt": "مسئله : تشخیص ارتباط , آیا متن دوم خلاصه ی متن اول است ؟",
        "task_classes": None,
    },
    "3_8": {
        "task_prompt": "مسئله : تشخیص ارتباط , آيا متن دوم بازنویسی متن اول است ؟",
        "task_classes": None,
    },
    "3_9": {
        "task_prompt": "مسئله : تشخیص ارتباط , آيا متن دوم ترجمه ی فارسی متن اول است ؟",
        "task_classes": None,
    },
    "3_10": {
        "task_prompt": "مسئله : تشخیص ارتباط , آيا متن دوم ترجمه ی عربی متن اول است ؟",
        "task_classes": None,
    },
    "3_11": {
        "task_prompt": "مسئله : تشخیص ارتباط , آيا متن دوم ترجمه ی انگلیسی متن اول است ؟",
        "task_classes": None,
    },
    "3_12": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول یک مکالمه است. آیا متن دوم خلاصه ی متن اول است ؟",
        "task_classes": None,
    },
    "3_13": {
        "task_prompt": "مسئله : تشخیص ارتباط , آیا متن دوم به متن اول مرتبط است ؟",
        "task_classes": None,
    },
    "3_14": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول مکالمه ی کاربر با چت بات است. آیا متن دوم موضوع استخراج شده ی متن اول است ؟",
        "task_classes": None,
    },
    "1_170": {
        "task_prompt": "مسئله : دسته بندی , دسته بندی موضوعی متن",
        "task_classes": [],
    },
    "1_190": {
        "task_prompt": "مسئله : دسته بندی , تحلیل احساس رضایت متن",
        "task_classes": ["مثبت", "منفی"],
    },
    "3_30": {
        "task_prompt": "مسئله : تشخیص ارتباط , متن اول  پیام جدید کاربر است. آیا متن دوم که یک سوال و پاسخ است به متن اول مرتبط است ؟",
        "task_classes": None,
    },
}


def preprocess_sample_easy_v7(
    sample, dataset_name, prompt_type, task_prompt_dict, dataset_info_dict, sub
):
    dataset_info = dataset_info_dict[dataset_name]
    task_id = dataset_info["task_id"]
    subtask_id = dataset_info["subtask_id"]
    if task_id != 4:
        task_key = str(task_id) + "_" + str(subtask_id)
        task_dict = task_prompt_dict[task_key]
        task_prompt = task_dict["task_prompt"]

    processed_sample = None
    if task_id == 1:
        processed_sample = task_prompt + " | " + "متن : " + sample

    elif task_id == 2:
        processed_sample = (
            task_prompt
            + " | "
            + "متن اول : "
            + sample[0]
            + " | "
            + "متن دوم : "
            + sample[1]
        )

    elif task_id == 3 and not sub:
        if prompt_type.value == "query":
            processed_sample = task_prompt + " | " + "متن اول : " + sample
        elif prompt_type.value == "passage":
            processed_sample = task_prompt + " | " + "متن دوم : " + sample

    elif task_id == 3 and sub:
        if sub == "sentence1":
            processed_sample = task_prompt + " | " + "متن اول : " + sample
        elif sub == "sentence2":
            processed_sample = task_prompt + " | " + "متن دوم : " + sample

    elif task_id == 4:
        processed_sample = sample
    return processed_sample


def preprocess_sample(sample, dataset_name, prompt_type, model_name, sub):
    processed_sample = preprocess_sample_easy_v7(
        sample, dataset_name, prompt_type, task_prompt_dict_v3, dataset_info_dict_2, sub
    )
    return processed_sample


headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


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
        print("\n\nyes you right...\n\n")
        print(task_name)
        print(type(prompt_type))
        print(len(sentences))
        print("before...")
        print(sentences[0])

        sub = kwargs.get("sub", None)
        print(f"\n sub === {sub}")

        sentences = [
            preprocess_sample(sentence, task_name, prompt_type, self.model_name, sub)
            for sentence in sentences
        ]
        print(len(sentences))

        # print(task_name)
        print("after...")
        print(sentences[0])
        # url = f"https://mcinext.ai/api/{self.model_name}"
        url = "https://mcinext.ai/api/hakim"

        data = {
            "model": "Hakim",
            "input": sentences,
            "encoding_format": "float",
            "add_special_tokens": True,
        }

        response = requests.post(url, headers=headers, json=data)

        embeddings = [item["embedding"] for item in response.json()["data"]]
        embeddings = np.array(embeddings)

        print(f"\n\n\n================================================\n{embeddings}")

        return embeddings


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
