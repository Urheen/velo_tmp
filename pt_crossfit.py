"""Prompt tuiing using single gpu"""
import os
import jax
import logging
import argparse
import optax
import jax.numpy as jnp
from flax.training import train_state
from flax import traverse_util
from tasks.crossfit_metrics import evaluate, METRICS
from optax import softmax_cross_entropy_with_integer_labels
from tqdm import tqdm


DATA_DIR = './crossfit/data/'


def trim_batch(
    input_ids,
    attention_mask,
    pad_token_id,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    mask = jnp.any(jax.lax.ne(input_ids, pad_token_id), axis=0)
    index = sum(mask)
    return input_ids[:, :index], attention_mask[:, :index]


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters

    parser.add_argument("--dataset_name", default="nlp_forest_single", required=True)
    parser.add_argument("--dataset_seed", default=42, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--model_name", default="t5-large", required=False)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

    # Preprocessing/decoding-related parameters
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--gen_early_stop', action='store_true')
    # parser.add_argument("--append_another_bos", action='store_true', default=False)  maybe this is for bart?

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dev_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--train_seed', default=42, type=int,
                        help="Seed for training")
    parser.add_argument('--logging_steps', default=100, type=int)
    parser.add_argument('--eval_steps', default=200, type=int)
    
    # if use Adam
    parser.add_argument("--optim", default="velo", type=str)
    parser.add_argument("--num_train_steps", default=3000, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")


    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    run(args=args, logger=logger)

def create_train_state(
        model, 
        num_train_steps
):

    class PTTrainState(train_state.TrainState):

        def apply_gradients(self, *, grads, is_velo, loss, **kwargs):
            if not is_velo:
                updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
            else:
                updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, extra_args={'loss': loss})
            new_params = optax.apply_updates(self.params, updates)
            return self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                **kwargs,
            )
        
    
        
    # use velo optimizer
    from learned_optimization.research.general_lopt import prefab
    tx = prefab.optax_lopt(num_train_steps)
    # only optimize the prompts
    from utils.train_utils import set_to_zero, multi_transform
    """tx = optax.adamw(
        learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-6, 
        # mask=decay_mask_fn
    )"""
    partition_optimizers = {'trainable': tx, 'frozen': set_to_zero()}
    param_partitions = traverse_util.path_aware_map(lambda path, _: 'trainable' if 'prompt' in path else 'frozen', model.params)
    tx = multi_transform(partition_optimizers, param_partitions)
    tx.init(model.params)

    return PTTrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
    )



def train(args, logger, model, tokenizer, train_data, dev_data, pad_token_id, decoder_start_id):

    rng = jax.random.PRNGKey(args.train_seed)
    dropout_rngs = jax.random.PRNGKey(args.train_seed + 1)
    
    train_dataset = train_data.load_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )
    train_loader = train_data.load_dataloader(train_dataset, batch_size=args.train_batch_size)

    @jax.jit
    def train_step(
        state: train_state.TrainState, 
        batch, 
        dropout_rng
    ):
        """Train model with an optimizer on a batch"""
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 2)
        
        def loss_fn(params):
            model_output = model.__call__(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                params=params, 
                dropout_rng=dropout_rng, 
                train=True
            )
            lm_logits = model_output.logits
            loss = softmax_cross_entropy_with_integer_labels(
                logits=lm_logits,
                labels=batch['labels']
            )
            """print(batch['input_ids'].shape)
            print(lm_logits.shape)
            print(batch['labels'].shape)
            print(loss.shape)
            print(batch['decoder_attention_mask'])
            
            exit()"""
            return jnp.sum(loss * batch['decoder_attention_mask']) / jnp.sum(batch['decoder_attention_mask'])

        loss, grad = jax.value_and_grad(loss_fn)(state.params)
        # if parrallele training
        # grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad, is_velo= args.optim == 'velo', loss=loss)
        metrics = {"loss": loss}
        return new_state, metrics, new_dropout_rng
    
    state = create_train_state(
        model=model,
        num_train_steps=args.num_train_steps
    )

    logger.info(f" ==== Start training ({args.num_train_steps} steps) ====")
    train_time = 0

    train_iterator = iter(train_loader)
    for step in tqdm(range(args.num_train_steps), desc=f"Epoch ... (0 / {args.num_train_steps})", position=0):
        batch = next(train_iterator)
        _batch = {}
        _batch['input_ids'], _batch['attention_mask'] = trim_batch(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            pad_token_id=pad_token_id
        )
        _batch['decoder_input_ids'], _batch['decoder_attention_mask'] = trim_batch(
            input_ids=batch['decoder_input_ids'], attention_mask=batch['decoder_attention_mask'],
            pad_token_id=pad_token_id
        )
        _batch['labels'], _ = trim_batch(
            input_ids=batch['labels'], attention_mask=batch['decoder_attention_mask'],
            pad_token_id=pad_token_id
        )
        _batch = batch
        state, train_metric, dropout_rngs = train_step(state, _batch, dropout_rngs)

        if step % args.logging_steps == 0:
            logger.info(f"Step {step} | Training Loss: {train_metric['loss']}")

        if step % args.eval_steps == 0 and step != 0:
            # evaluation
            curr_performance = inference(
                logger,
                model=model,
                params=state.params,
                tokenizer=tokenizer,
                dev_data=dev_data,
                args=args,
                decoder_start_id=decoder_start_id
            )
            logger.info(f"Step {step} | {dev_data.metric}: {curr_performance} on epoch=%d")


def inference(
    logger,
    model,
    params,
    tokenizer, 
    dev_data,
    args, 
    decoder_start_id,
    save_predictions=True
):
    dev_dataset = dev_data.load_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )
    dev_loader = dev_data.load_dataloader(dev_dataset, batch_size=args.dev_batch_size)
    model.params = params
    @jax.jit
    def generate(input_ids, attention_mask):
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            num_beams=args.num_beams,
            max_length=args.max_output_length,
            # decoder_start_id=model.config.decoder_start_token_id,
            early_stopping=args.gen_early_stop
        )
        return outputs
    
    predictions = []
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dev_loader)):
        """input_ids, attention_mask = trim_batch(
            batch['input_ids'], 
            batch['attention_mask'], 
            tokenizer.pad_token_id
        )
        outputs = generate(            
            input_ids=input_ids, 
            attention_mask=attention_mask, 

        )"""
        input_ids, attention_mask = batch['input_ids'], batch["attention_mask"]
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            num_beams=args.num_beams,
            max_length=args.max_output_length,
            # decoder_start_id=model.config.decoder_start_token_id,
            early_stopping=args.gen_early_stop
        )
        summary_ids = jax.device_get(outputs.sequences)
        for i in range(len(summary_ids)):
            predictions.append(tokenizer.decode(
                summary_ids[i],
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True))
            
    if save_predictions:
        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(args.output_dir, "{}_predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            f.writelines(prediction_text)
    logger.info("Saved prediction in {}".format(save_path))

    return dev_data.evaluate(predictions)
        

def get_data_path(dataset_name, dataset_seed, split):
    file_names = os.listdir(os.path.join(DATA_DIR, dataset_name))
    for fn in file_names:
        if split in fn and dataset_name in fn and str(dataset_seed) in fn:
            return os.path.join(DATA_DIR, dataset_name, fn)
        

def run(args, logger):
    from transformers import T5Tokenizer, T5Config
    from tasks.crossfit_dataset import CrossfitSingleTaskData
    from models.prompt_t5 import FlaxPromptT5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    config = T5Config.from_pretrained(args.model_name)

    # 1. load data
    train_data = CrossfitSingleTaskData(
        data_path=get_data_path(args.dataset_name, args.dataset_seed, 'train')
    )

    dev_data = CrossfitSingleTaskData(
        data_path=get_data_path(args.dataset_name, args.dataset_seed, 'test')
    )

    # 2. training
    model = FlaxPromptT5ForConditionalGeneration.from_pretrained(args.model_name)
    train(args, logger, model, tokenizer, train_data, dev_data, tokenizer.pad_token_id, config.decoder_start_token_id)


if __name__ == '__main__':
    main()