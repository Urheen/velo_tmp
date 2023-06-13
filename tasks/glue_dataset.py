import jax
import numpy as np
from transformers import PretrainedConfig, AutoConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from flax.training.common_utils import shard
from learned_optimization.tasks.datasets import base
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

import logging
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": f"The name of the glue task to train on. choices {list(task_to_keys.keys())}"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )


    def __post_init__(self):
        if self.task_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower() if type(self.task_name) == str else self.task_name

def glue_train_data_collator(rng, dataset, batch_size):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    i = 0
    while True:
        perm = perms[i%len(perms)]
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        # batch = shard(batch)

        yield batch

def _make_dataset(model_args, data_args, tokenizer):
    
    print(f"Load raw dataset from {data_args.task_name}")
    raw_datasets = load_dataset(
        "glue",
        data_args.task_name,
        use_auth_token=True if model_args.use_auth_token else None,
        cache_dir='./datasets'
    )

    # labels:
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets['train'].features['label'].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        use_auth_token=True if model_args.use_auth_token else None,
        cache_dir='/home/huangzeyu/movelo/velo_peft/hf_models/'
    )
    

    # dataset preprocessing
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # some models have set the order of the labels to use, so let's make sure we do use it
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # some have all caps in their cofig, some don't
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    rng = jax.random.PRNGKey(0)
    rng, input_rng = jax.random.split(rng)
    train_loader = glue_train_data_collator(
        input_rng, 
        train_dataset, 
        # train_batch_size
        16
    )
    return base.Datasets(
        train=train_loader,
        inner_valid=train_loader,
        outer_valid=train_loader,
        test=train_loader
    ), raw_datasets, is_regression, num_labels, config


def glue_datasets(model_args, data_args, tokenizer):
    return _make_dataset(model_args, data_args, tokenizer)