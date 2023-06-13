DEBUG_SOURCE = ["glue-mrpc", "glue-sst2"]
DEBUG_TARGET = ["glue-qnli"]



RANDOM_SOURCE = [
    "glue-mrpc", "math_qa", "quarel", "e2e_nlg_cleaned", "tweet_eval-stance_atheism", 
    "lama-squad", "tab_fact", "aqua_rat", "tweet_eval-emoji", "glue-wnli", "codah", 
    "tweet_eval-offensive", "wiki_qa", "blimp-ellipsis_n_bar_1", "openbookqa", "sms_spam", 
    "acronym_identification", "blimp-determiner_noun_agreement_with_adj_irregular_1", 
    "ethos-national_origin", "spider", "definite_pronoun_resolution", "hellaswag", 
    "superglue-wsc", "numer_sense", "ade_corpus_v2-dosage", "blimp-ellipsis_n_bar_2", 
    "kilt_ay2", "squad-no_context", "google_wellformed_query", "xsum", "wiqa", 
    "tweet_eval-stance_abortion", "reddit_tifu-tldr", "ade_corpus_v2-effect", "qa_srl", 
    "ethos-religion", "commonsense_qa", 
    # "jeopardy",  # jeopardy not found in Qin et al.
    "biomrc", "superglue-multirc", "ethos-race", "eli5-askh", "glue-qqp", "paws", 
    "ethos-directed_vs_generalized", "glue-sst2", 
    # "mocha", # mocha not found in Qin et al.
    "tweet_eval-hate", "glue-rte", "blimp-anaphor_number_agreement", 
    "lama-conceptnet", "hate_speech_offensive", "superglue-wic", "boolq", "kilt_hotpotqa", 
    "quartz-no_knowledge", "aslg_pc12", "sick", "tweet_eval-stance_climate", 
    "tweet_eval-sentiment", "crows_pairs", "glue-mnli", "medical_questions_pairs", 
    "break-QDMR-high-level", "qasc", "imdb", "ethos-gender", "trec-finegrained", 
    "adversarialqa", "onestop_english", "web_questions", "duorc", 
    "yelp_review_full", "swag", "proto_qa", "scitail", "tweet_eval-stance_feminist", "limit", 
    "common_gen", "scicite", "blimp-irregular_past_participle_adjectives", "social_i_qa", 
    "anli", "kilt_zsre", "cosmos_qa", "superglue-record", "squad-with_context", "emotion", 
    "blimp-existential_there_quantifiers_1", "race-middle", "kilt_wow", "sciq", "wino_grande", 
    "rotten_tomatoes", "superglue-cb", "poem_sentiment", "ropes", "reddit_tifu-title", "piqa", 
    "climate_fever", "lama-google_re", "search_qa", 
    # "wiki_auto", # not found in Qin et al. 
    "mc_taco", "blimp-wh_questions_object_gap", "hotpot_qa", "emo", "kilt_nq", "kilt_trex", 
    "quartz-with_knowledge", "dbpedia_14", "yahoo_answers_topics", 
    # "app_reviews", # not dounf in Qin et al.
    "superglue-copa", "blimp-anaphor_gender_agreement", "hate_speech18", "gigaword", 
    "multi_news", "aeslc", "quail"
]  # 116 in total

RANDOM_TARGET = [
    "quoref", "wiki_split", "ethos-disability", "yelp_polarity", "superglue-rte", 
    "glue-cola", "ethos-sexual_orientation", "blimp-sentential_negation_npi_scope", 
    "ai2_arc", "amazon_polarity", "race-high", "blimp-sentential_negation_npi_licensor_present", 
    "tweet_eval-irony", 
    # "break-QDMR", # not in Qin et al.
    "crawl_domain", "freebase_qa", "glue-qnli", "hatexplain", "ag_news", "circa",
    "sam-sum"  # not in original crossfit
]  # 20 in total

CLASSIFICATION_SOURCE = [
    "superglue-rte", "tweet_eval-sentiment", "discovery", "glue-rte", "superglue-wsc", 
    "scicite", "glue-mrpc", "tweet_eval-stance_hillary", "tweet_eval-offensive", "emotion", 
    "hatexplain", "glue-cola", "sick", "paws", "ethos-sexual_orientation", "glue-qqp", 
    "tweet_eval-emotion", "sms_spam", "health_fact", "glue-mnli", "imdb", "ethos-disability", 
    "glue-wnli", "scitail", "trec-finegrained", "yahoo_answers_topics", "liar", 
    "glue-sst2", "tweet_eval-stance_abortion", "circa", "tweet_eval-stance_climate", 
    "glue-qnli", "tweet_eval-emoji", "ethos-directed_vs_generalized", 
    "ade_corpus_v2-classification", 
    "ag_news" # not in original crossfit
    # "wiki_auto", # not in Qin et al.
    "hate_speech_offensive", "superglue-wic", "google_wellformed_query", 
    "tweet_eval-irony", "ethos-gender", "onestop_english", "trec", "rotten_tomatoes", 
    "kilt_fever"
]


CLASSIFICATION_TARGET = [
    "superglue-cb", "dbpedia_14", "wiki_qa", "emo", "yelp_polarity", "ethos-religion", 
    # "financial_phrasebank", not found in Qin et al. 
    "tab_fact", "anli", "ethos-race"]


NON_CLASSIFICATION_SOURCE = [
    "ade_corpus_v2-dosage", "art", "biomrc", 
    "blimp-anaphor_number_agreement", "blimp-ellipsis_n_bar_2", 
    "blimp-sentential_negation_npi_licensor_present", 
    "blimp-sentential_negation_npi_scope", "break-QDMR-high-level", 
    "commonsense_qa", "crows_pairs", "dream", "duorc", "eli5-asks", "eli5-eli5", 
    "freebase_qa", "gigaword", "hellaswag", "hotpot_qa", "kilt_ay2", 
    "kilt_hotpotqa", "kilt_trex", "kilt_zsre", "lama-conceptnet", "lama-google_re", 
    "lama-squad", "math_qa", "numer_sense", "openbookqa", "piqa", "proto_qa", "qa_srl", 
    "quarel", "quartz-no_knowledge", "race-high", "reddit_tifu-title", "reddit_tifu-tldr", 
    "ropes", "sciq", "social_i_qa", "spider", "superglue-multirc", "wiki_bio", "wikisql", 
    "xsum", "yelp_review_full"]


NON_CLASSIFICATION_TARGET = [
    "multi_news", "superglue-copa", "quail", "blimp-anaphor_gender_agreement", 
    "common_gen", "acronym_identification", "quoref", "wiki_split", "ai2_arc", 
    "break-QDMR", "crawl_domain", "samsum"
]


BOTH_SOURCE = [
    "ade_corpus_v2-dosage", "biomrc", "blimp-ellipsis_n_bar_2", 
    "blimp-sentential_negation_npi_scope", "commonsense_qa", "crows_pairs", "duorc", 
    "hellaswag", "kilt_zsre", "lama-google_re", "lama-squad", "math_qa", "numer_sense", 
    "openbookqa", "piqa", "proto_qa", "quartz-no_knowledge", "race-high", 
    "reddit_tifu-tldr", "ropes", "sciq", "wiki_bio", "discovery", "emotion", 
    "ethos-disability", "ethos-sexual_orientation", "glue-cola", "glue-mnli", "glue-mrpc", 
    "glue-qqp", "glue-rte", "glue-wnli", "hatexplain", "health_fact", "imdb", "paws", 
    "scicite", "sick", "sms_spam", "superglue-rte", "superglue-wsc", "tweet_eval-emotion", 
    "tweet_eval-offensive", "tweet_eval-sentiment", "tweet_eval-stance_hillary"
]

QA_SOURCE = [
    "biomrc", "boolq", "freebase_qa", "hotpot_qa", "kilt_hotpotqa", "kilt_nq", 
    "kilt_trex", "kilt_zsre", "lama-conceptnet", "lama-google_re", "lama-squad", 
    "lama-trex", "mc_taco", "numer_sense", "quoref", "ropes", "search_qa", "squad-no_context", 
    "superglue-multirc", "superglue-record", "tweet_qa", "web_questions"
]

"""
    "adversarialqa", # not in Qin et al.
    "biomrc", "boolq",
    "duorc", "eli5-askh", "eli5-asks", "eli5-eli5",  # not in Qin et al.
    "freebase_qa", "hotpot_qa",
    "jeopardy", # not in Qin et al.
    "kilt_hotpotqa", "kilt_nq", "kilt_trex", "kilt_zsre",
    "lama-conceptnet", "lama-google_re", "lama-squad", "lama-trex",
    "mc_taco", "numer_sense", "quoref", "ropes", "search_qa",
    "squad-no_context",
    "squad-with_context",  # not in Qin et al.
    "superglue-multirc", "superglue-record", "tweet_qa",
    "web_questions"
"""

QA_TARGET =  [
    "ai2_arc",
    # "aqua_rat", not in Qin et al.
    "codah",
    # "commonsense_qa", not in Qin et al.
    "cosmos_qa", "dream", "hellaswag",
    # "math_qa", "openbookqa", not in Qin et al.
    "qasc", "quail", "quarel", "quartz-no_knowledge",
    "quartz-with_knowledge",
    # "race-high", "race-middle", not in Qin et al.
    "sciq",
    # "social_i_qa", not in Qin et al.
    "superglue-copa", "swag", "wino_grande", "wiqa"
]


NON_PARAPHRASE_CLASSIFICATION_SOURCE = [
    "ade_corpus_v2-classification", "ag_news", "amazon_polarity",
    "anli", # not in the original crossfit
    "circa", "climate_fever", "dbpedia_14", "discovery", "emo",
    "emotion", "ethos-directed_vs_generalized", "ethos-disability",
    "ethos-gender", "ethos-national_origin", "ethos-race", "ethos-religion",
    "ethos-sexual_orientation", "financial_phrasebank",
    "glue-cola", "glue-mnli", "glue-rte",
    # "glue-mrpc", "glue-qqp",  not in Qin et al.
    "glue-sst2",
    "glue-mnli"  # not in the original crossfit
    "google_wellformed_query", "hate_speech18", "hate_speech_offensive",
    "hatexplain", "health_fact", "imdb", "kilt_fever", "liar",
    # "medical_questions_pairs", not in Qin et al.
    "onestop_english",
    # "paws", not in Qin et al.
    "poem_sentiment", "rotten_tomatoes", "scicite", "sick", "sms_spam",
    "superglue-cb", "superglue-rte" # not in the original crossfit
    "superglue-wic", "superglue-wsc", "tab_fact", "trec", "trec-finegrained",
    "tweet_eval-emoji", "tweet_eval-emotion", "tweet_eval-hate",
    "tweet_eval-irony", "tweet_eval-offensive", "tweet_eval-sentiment",
    "tweet_eval-stance_abortion", "tweet_eval-stance_atheism", "tweet_eval-stance_climate",
    "tweet_eval-stance_feminist", "tweet_eval-stance_hillary",
    # "wiki_auto", not in Qin et al.
    "wiki_qa", "yahoo_answers_topics", "yelp_polarity"
]


PARAPHRASE_TARGET = ["glue-mrpc", "glue-qqp", "medical_questions_pairs", "paws"]


# partition not in the crossfit
NON_QA_SOURCE = [
    "hate_speech_offensive", "google_wellformed_query", "circa", "glue-sst2", 
    "scitail", "emo", "ag_news", "art", "paws", "kilt_ay2", "glue-qnli", 
    "ade_corpus_v2-classification", "hatexplain", "emotion", "glue-qqp", 
    "kilt_fever", "dbpedia_14", "glue-mnli", "discovery", "gigaword", "amazon_polarity", 
    "tab_fact", "tweet_eval-emoji", "tweet_eval-offensive", "tweet_evalsentiment", 
    "imdb", "liar", "anli", "wikisql", "xsum", "yahoo_answers_topics", "yelp_polarity", 
    "yelp_review_full"
]

SOURCES = {
    'random': RANDOM_SOURCE,
    'classification': CLASSIFICATION_SOURCE,
    'non_classification': NON_CLASSIFICATION_SOURCE,
    'both': BOTH_SOURCE,
    'qa': QA_SOURCE,
    'non_paraphrase': NON_PARAPHRASE_CLASSIFICATION_SOURCE,
    'non_qa': NON_QA_SOURCE,
    'debug': DEBUG_SOURCE
}
    

TARGETS = {
    'random': RANDOM_TARGET,
    'classification': CLASSIFICATION_TARGET,
    'non_classification': NON_CLASSIFICATION_TARGET,
    'qa': QA_TARGET,
    'paraphrase': PARAPHRASE_TARGET,
    'debug': DEBUG_SOURCE
}








