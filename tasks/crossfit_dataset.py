import os
import jax
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
from transformers import T5Tokenizer, T5Config
from learned_optimization.tasks.datasets import base
from tasks.crossfit_metrics import METRICS, evaluate

# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    input_ids = jnp.asarray(input_ids)
    shifted_input_ids = jnp.asarray(input_ids)
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids.tolist()


class CrossfitSingleTaskData(object):

    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.is_train = 'train' in data_path
        self.task_name = "_".join(self.data_path.split("/")[-1].split("_")[:-3])
        with open(data_path) as data_file:
            lines = data_file.readlines()

        # train_examples = []
        self.data = []
        for line in lines:
            d = line.strip().split("\t")
            self.data.append((d[0], d[1:]))

        self.metric = METRICS[self.task_name]

    def __len__(self):
        return len(self.data)
    
    def decode(self, tokens, tokenizer):
        return tokenizer.decode(
            tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(
            self, 
            tokenizer: T5Tokenizer,
            decoder_start_id: int,
            max_input_length: int ,
            max_output_length: int,
            do_lowercase: bool = True,
            append_another_bos: bool = False
        ):
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".tsv", f"-{postfix}.json")
        
        )

        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)
        else:
            inputs = []
            outputs = []

            for dp in self.data:
               # add t5-style prefix
               inputs.append(f"[{self.task_name}] {dp[0]}")
               outputs.append(dp[1])
            
            outputs, metadata = self.flatten(outputs)

            if do_lowercase:
                inputs = [inp.lower() for inp in inputs]
                outputs = [out.lower() for out in outputs]
            if append_another_bos:
                inputs = ["<s> " + inp for inp in inputs]
                outputs = ["<s> " + out for out in outputs]

            tokenized_input = tokenizer.batch_encode_plus(
                inputs,
                pad_to_max_length=True,
                max_length=max_input_length
            )      
            tokenized_output = tokenizer.batch_encode_plus(
                outputs,
                pad_to_max_length=True,
                max_length=max_output_length
            )
            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            labels, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
            decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id, decoder_start_id)
            # save the preprocessed data
            """
            with open(preprocessed_path, 'w') as f:
                json.dump(
                    [
                        input_ids, attention_mask,
                        decoder_input_ids, decoder_attention_mask,
                        metadata
                    ], f
                )
            """
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "input_ids": input_ids, 
                "attention_mask": attention_mask,
                "labels": labels,
                "decoder_input_ids": decoder_input_ids, 
                "decoder_attention_mask": decoder_attention_mask

            }
        )
        return ds
    
    def load_dataloader(self, ds, batch_size):
        if self.is_train:
            ds = ds.repeat()
        ds = ds.shuffle(batch_size * 1000)
        ds = ds.batch(batch_size, drop_remainder=self.is_train)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = tfds.as_numpy(ds)
        return ds
    
    def evaluate(self, predictions):
        assert len(predictions) == len(self)
        predictions = [prediction.strip() for prediction in predictions]
        return evaluate(predictions, self.data, self.metric)


def _make_dataset(
    data_name: str,
    seed: int,
    tokenizer_name: str,
    max_input_length = 256,
    max_output_length = 32,
    batch_size = 2
):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    config = T5Config.from_pretrained(tokenizer_name)
    def make(path):
        def iterator_fn():
            data = CrossfitSingleTaskData(path)
            dataset = data.load_dataset(
                tokenizer=tokenizer,
                decoder_start_id=config.decoder_start_token_id,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
            )
            data_loader = data.load_dataloader(ds=dataset, batch_size=batch_size)
            return iter(data_loader)
        return base.ThreadSafeIterator(base.LazyIterator(iterator_fn))

    f_names = os.listdir(f'./crossfit/data/{data_name}/')
    paths = []
    for n in f_names:
        paths.append(os.path.join(
            f'./crossfit/data/{data_name}/', n
        ))

    data_paths = {}
    for p in paths:
        if 'train' in p and str(seed) in p:
            data_paths['train'] = p
        if 'dev' in p and str(seed) in p:
            data_paths['dev'] = p
        if 'test' in p and str(seed) in p:
            data_paths['test'] = p

    train = make(data_paths['train']) if 'train' in data_paths else None
    valid = make(data_paths['dev']) if 'dev' in data_paths else None
    test = make(data_paths['test']) if 'test' in data_paths else None
    return base.Datasets(
        train=train, 
        inner_valid=valid,
        outer_valid=valid, 
        test=test
    )
        


             