'''The script that used to fine-tune velo on single'''
import jax
import numpy as np
import tqdm
import functools
from transformers import (HfArgumentParser, AutoTokenizer)
from learned_optimization.optimizers import base as opt_base
from dataclasses import dataclass, field
from typing import Optional
from loptim_pt import PartialHyperV2
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base
import os
import tensorflow as tf
# os.environ['TFDS_DATA_DIR'] = '/home/huangzeyu/tensorflow_datasets'
tf.config.experimental.set_visible_devices([], "GPU")

def main():
    # from models.prompt_bert import ModelArguments as BertModelArguments
    # from models.prompt_t5 import FlaxPromptT5ForConditionalGeneration
    # from tasks.glue_dataset import DataTrainingArguments as GlueTrainingArguments
    #parser = HfArgumentParser(
    #    (BertModelArguments, GlueTrainingArguments, MetaFineTuningArguments)
    #)
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # from tasks.glue_pbert import promptbert_glue
    # task = promptbert_glue(model_args, data_args)
    import argparse
    parser = argparse.ArgumentParser()
    "Hyper-parameters for learned optimization meta-training"
    parser.add_argument('--meta_opt', default='adam', type=str)
    parser.add_argument('--outer_learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_partitions', default='debug', type=str)
    parser.add_argument('--accumulate_averages', default=20, type=int)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()
    train(args)


def train(training_args):
    # 1. Defined the learned optimizer
    update_params = ('prompt',)
    lopt = PartialHyperV2(
        update_params=update_params,
        use_bugged_loss_features=False,
        param_inits=256,
        lstm_hidden_size=512
    )

    # 2. Define the meta optimizer
    from learned_optimization.outer_trainers.gradient_learner import GradientLearner
    from learned_optimization.optimizers import Adam, GradientClipOptimizer
    from learned_optimization.optimizers.gradient_accumulator import GradientAccumulator
    theta_opt = Adam(learning_rate=training_args.outer_learning_rate)
    theta_opt = GradientAccumulator(
        opt=theta_opt, 
        num_average=training_args.accumulate_averages
    )
    theta_opt = GradientClipOptimizer(opt=theta_opt)
        
    from utils.outer_train import run_train, build_gradient_estimators
    from learned_optimization.outer_trainers.full_es import FullES
    from learned_optimization.outer_trainers.truncated_es import TruncatedES
    from learned_optimization.outer_trainers.truncation_schedule import (
        LogUniformLengthSchedule,
        ConstantTruncationSchedule,
        NeverEndingTruncationSchedule
    )
    from learned_optimization.outer_trainers.lopt_truncated_step import VectorizedLOptTruncatedStep
    from tasks.crossfit_pt5 import sample_crossfit_T5_base
    sample_task_family_fn = functools.partial(
        sample_crossfit_T5_base,
        partitions=training_args.train_partitions
    )

    # define outer trainer function
    def outer_trainer_fn():
        return GradientLearner(
        init_theta_from_path='./pretrained_lopts/aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620/params',
        reset_outer_iteration=True,
        theta_opt=theta_opt,
        meta_init=lopt
    )

    def truncated_step_fn(task_family, lopt):
        return VectorizedLOptTruncatedStep(
            task_family=task_family, 
            learned_opt=lopt, 
            # trunc_sched=trunc_sched,
            trunc_sched=NeverEndingTruncationSchedule(),
            random_initial_iteration_offset=0,
            num_tasks=4 # soueced from large_scale_phase4
        )
    
    def gradient_estimator_fn(trunc_step):
        return FullES(
            truncated_step=trunc_step,
            truncation_schedule=LogUniformLengthSchedule(min_length=20, max_length=2000),
            loss_type="last_recompute",  # avg, min or last_recompute
            recompute_samples=100,
            sign_delta_loss_scalar=1.0
        )
    
    
    # from learned_optimization.outer_trainers.truncated_es
    os.makedirs(training_args.output_dir)
    run_train(
        train_log_dir=training_args.output_dir,
        lopt=lopt,  
        # the learned optimizer
        outer_learner_fn=outer_trainer_fn,  
        # fn which produces the learner which does the actual training of the lopt weights.
        num_estimators=2,# num of estimators to use per outer update
        is_trainer=True, # to run a trainer / learner
        is_worker=True, # to run a worker
        num_steps=1000,
        trainer_batch_size=8,  # 512 in large_scale_phase4
        # size of meta-gradients / number of different gradient estimates to aggreate over
        staleness=500,
        # how stale gradients can bee before throwing them fout.
        stochastic_resample_frequency=200,
        summary_every_n=25,
        sample_estimator_fn=functools.partial(
            build_gradient_estimators,
            sample_task_family_fn=sample_task_family_fn,
            gradient_estimator_fn=gradient_estimator_fn,
            truncated_step_fn=truncated_step_fn
        ),
        num_workers=None,
        learner_mode="async",
        run_num_estimators_per_gradient=1 
        # send gradients up every step, sourced from the latrge_scale_phase4
    )


if __name__ == '__main__':
    main()
    """
    Please try the following tasks later by running individual files: 
    ['samsum.py', 'search_qa.py', 'squad.py', 'swag.py', 'tab_fact.py', 'trec.py', 'trec_finegrained.py', 
     'tweet_eval.py', 'cosmos_qa.py', 'crawl_domain.py', 'crows_pairs.py', 'dbpedia_14.py', 'discovery.py', 
     'ethos.py', 'financial_phrasebank.py', 'freebase_qa.py', 'gigaword.py', 'reddit_tifu.py', 'piqa.py', 
     'poem_sentiment.py', 'proto_qa.py', 'quail.py', 'agnews.py', 'amazon_polarity.py', 'app_reviews.py', 
     'aqua_rat.py', 'acronym_identification.py', 'ade_classification.py', 'ade_dosage.py', 'ade_effect.py', 
     'adversarial_qa.py', 'google_wellformed_query.py', 'hate_speech_offensive.py', 'hatexplain.py', 'dream.py', 
     'duorc.py', 'e2e_nlg_cleaned.py', 'emo.py', 'blimp.py', 'break.py', 'health_fact.py', 'hellaswag.py', 'hotpot_qa.py', 
     'imdb.py', 'jeopardy.py', 'circa.py', 'climate_fever.py', 'codah.py', 'cos_e.py', 'mocha.py', 'multi_news.py', 'numer_sense.py', 
     'onestop_english.py', 'limit.py', 'math_qa.py', 'mc_taco.py', 'medical_questions_pairs.py', 'wiki_auto.py', 'wiki_split.py', 
     'wikisql.py', 'xsum.py', 'yahoo_answers_topics.py', 'yelp_polarity.py', 'yelp_review_full.py']
    """