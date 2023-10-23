from flask import Flask, jsonify, request

import torch
import logging
import os
import sys
import glob
import json
from dataclasses import dataclass, field
from typing import Optional
import tempfile
from pathlib import Path


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EncoderDecoderModel,
)
from transformers.trainer_utils import EvaluationStrategy, IntervalStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    MultiDataset,
    TokenizedDataset,
    TokenizedDataCollator,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler


logger = logging.getLogger(__name__)

trainer, tokenizer, data_args = None, None, None
app = Flask(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={
                                 "help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={
                                "help": "Whether  to freeze the embeddings."})
    is_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "is encoder decoder"},
    )
    tie_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "tie encoder decoder"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={
            "help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(
        default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(
        default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(
        default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(
        default=4, metadata={"help": "# num_beams to use for evaluation."})
    length_penalty: Optional[float] = field(
        default=0.6, metadata={"help": "# length_penalty"})
    no_repeat_ngram_size: Optional[int] = field(
        default=None, metadata={"help": "# num_beams to use for evaluation."})
    upsampling_factor: Optional[float] = field(default=None,
                                               metadata={
                                                   "help": "# use data upsampling factor only when using multiple data files."}
                                               )
    rouge_lang: Optional[str] = field(default=None,
                                      metadata={
                                          "help": "# apply language specific tokenization and stemming (if available)"}
                                      )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


def load_model():
    global tokenizer, trainer, data_args
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    # training_args.local_rank = -1
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {}
    if data_args.max_target_length:
        config_kwargs.update({'max_length': data_args.max_target_length})
    if data_args.eval_beams:
        config_kwargs.update({'num_beams': data_args.eval_beams})
    if data_args.length_penalty:
        config_kwargs.update({'length_penalty': data_args.length_penalty})
    if data_args.no_repeat_ngram_size:
        config_kwargs.update(
            {'no_repeat_ngram_size': data_args.no_repeat_ngram_size})

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, **config_kwargs,
    )

    extra_model_params = ("encoder_layerdrop",
                          "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(
                config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False, cache_dir=model_args.cache_dir,
    )

    if model_args.is_encoder_decoder:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_args.model_name_or_path,
                                                                    model_args.model_name_or_path, tie_encoder_decoder=model_args.tie_encoder_decoder,
                                                                    cache_dir=model_args.cache_dir
                                                                    )
        # just to be safe
        tokenizer.bos_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        tokenizer.eos_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.update(config_kwargs)
        model.config.encoder.update(config_kwargs)
        model.config.decoder.update(config_kwargs)
        model.config.decoder.is_decoder = True
        model.config.decoder.add_cross_attention = True
        config = model.config

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if training_args.local_rank != -1 else 1)
    )
    logger.info(f'Effective batch size: {total_train_batch_size}')
    preprocessed_dir = os.path.join(training_args.output_dir, "preprocessed")

    dataset_class = Seq2SeqDataset
    dataset_collator = Seq2SeqDataCollator(
        tokenizer, data_args, None, training_args.tpu_num_cores)

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(
            data_args.task, tokenizer, data_args) if training_args.predict_with_generate else None
    )
    training_args.remove_unused_columns = False
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=dataset_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )
    logger.info(f"Model {model_args.model_name_or_path} successfully loaded...")


def predict(data_dir):
    dataset_class = Seq2SeqDataset
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix="",
        )
    )

    test_output = trainer.predict(
        test_dataset=test_dataset,
        metric_key_prefix="test",
        max_length=data_args.val_max_target_length,
        num_beams=data_args.eval_beams,
    )

    predictions = test_output.predictions
    predictions[predictions == -100] = tokenizer.pad_token_id
    test_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    test_preds = lmap(str.strip, test_preds)

    return test_preds


@app.route("/", methods=['POST'])
def do_prediction():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir.name
        tmpdirpath = Path(tmpdirname)
        data = request.get_data(as_text=True)
        logger.info("Number of lines in data: " + str(data.count('\n')))
        (tmpdirpath / "test.source").write_text(data, encoding="utf-8")
        (tmpdirpath / "test.target").symlink_to(tmpdirpath / "test.source")
        res = predict(tmpdirname)
        return '\n'.join(res)


if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=4123)
