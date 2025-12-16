#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune a Transformers model on SQuAD with Galore optimizer integration.

"""
import evaluate
import argparse
import collections
import logging
import os
import math
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from galore_torch import GaLoreAdamW, HMTAdamW, ApolloAdamW, Lotus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformers model on SQuAD with Galore optimizer integration"
    )
    # 模型及输出
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store the final model.")
    # 数据文件（SQuAD 格式 json），若不传，则自动加载 squad 数据集
    parser.add_argument("--train_file", type=str, default=None,
                        help="A json file containing the training data in SQuAD format.")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="A json file containing the validation data in SQuAD format.")
    # Tokenizer 参数
    parser.add_argument("--max_seq_length", type=int, default=384,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="When splitting a long document into chunks, how much stride to take between chunks.")
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="Batch size per device during evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay if applied.")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X update steps.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every X update steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X update steps.")
    # Galore / LoRA 参数
    parser.add_argument("--enable_galore", action="store_true",
                        help="Whether to enable Galore optimizer (low-rank optimization).")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank for low-rank adaptation.")
    parser.add_argument("--update_proj_gap", type=int, default=50,
                        help="Update projection gap parameter for Galore optimizer.")
    parser.add_argument("--galore_scale", type=float, default=1.0,
                        help="Scale parameter for Galore optimizer.")
    parser.add_argument("--proj_type", type=str, default="std",
                        help="Projection type for Galore optimizer.")
    parser.add_argument("--lora_all_modules", action="store_true",
                        help="Whether to apply LoRA to all modules.")
    parser.add_argument("--optimizer_name", type=str, default=None, choices=["Galore", "HMT", "Apollo", "Lotus"],
                        help="The name of the optimizer to use when enabling Galore.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Drift threshold for Galore optimizer.")
    parser.add_argument("--check_interval", type=int, default=50,
                        help="Check interval for Galore optimizer.")
    parser.add_argument("--proj_method", type=str, default="traditional", choices=["traditional", "iteration"],
                        help="Projection method for Galore optimizer.")
    parser.add_argument("--use_stable", action="store_true",
                        help="Whether to use stable subspace update in Galore optimizer.")

    args = parser.parse_args()
    return args


def prepare_train_features(examples, tokenizer, args):
    """
    对输入样本进行分词，并利用滑动窗口（overflow）获得多个 feature，
    同时保留 offsets、example_id、以及答案的起始和结束 token 索引。
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def post_process_qa_predictions(examples, features, predictions, tokenizer, args, n_best_size=20, max_answer_length=30):
    """
    基于预测的 start_logits 和 end_logits，后处理得到最终答案文本。
    """
    all_start_logits, all_end_logits = predictions
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    predicted_answers = {}
    for example in tqdm(examples, desc="Post-processing"):
        example_id = example["id"]
        context = example["context"]
        feature_indices = features_per_example[example_id]
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1].tolist()
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                        continue
                    prelim_predictions.append({
                        "start_index": start_index,
                        "end_index": end_index,
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                        "score": start_logits[start_index] + end_logits[end_index],
                        "offsets": (offsets[start_index], offsets[end_index])
                    })
        if prelim_predictions:
            best_pred = max(prelim_predictions, key=lambda x: x["score"])
            start_char = best_pred["offsets"][0][0]
            end_char = best_pred["offsets"][1][1]
            predicted_answers[example_id] = context[start_char: end_char]
        else:
            predicted_answers[example_id] = ""

    formatted_predictions = [{"id": ex["id"], "prediction_text": predicted_answers.get(ex["id"], "")} for ex in examples]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    squad_metric = evaluate.load("squad")
    return squad_metric.compute(predictions=formatted_predictions, references=references)


class CustomTrainer(Trainer):
    """
    Trainer 的子类，支持在 create_optimizer 中插入自定义 Galore 优化器。
    这里传入额外参数 custom_args（即 argparse 解析结果）以及原始 eval_examples 用于后处理。
    """
    def __init__(self, *args, custom_args=None, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args
        self.eval_examples = eval_examples

    def create_optimizer(self):
        if self.optimizer is None:
            if not self.custom_args.enable_galore:
                # 不启用 Galore 时，使用默认 AdamW 优化器
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
            else:
                import torch.nn as nn
                # 根据模型名称决定目标模块（例如 bert 模型使用 "query"）
                if not self.custom_args.lora_all_modules:
                    if "bert" in self.custom_args.model_name_or_path.lower():
                        target_modules_list = ["query"]
                    else:
                        target_modules_list = ["q_proj", "v_proj"]
                else:
                    if "bert" in self.custom_args.model_name_or_path.lower():
                        target_modules_list = ["query", "value", "key", "intermediate.dense", "output.dense"]
                    else:
                        target_modules_list = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj"]

                galore_params = []
                for module_name, module in self.model.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue
                    if not any(target_key in module_name for target_key in target_modules_list):
                        continue
                    print("Enabling Galore on module:", module_name)
                    galore_params.append(module.weight)
                id_galore_params = [id(p) for p in galore_params]
                regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params]
                param_groups = [
                    {'params': regular_params},
                    {'params': galore_params,
                     'rank': self.custom_args.lora_r,
                     'update_proj_gap': self.custom_args.update_proj_gap,
                     'scale': self.custom_args.galore_scale,
                     'proj_type': self.custom_args.proj_type,
                     'use_stable': self.custom_args.use_stable,
                     'drift_threshold': self.custom_args.threshold,
                     'check_interval': self.custom_args.check_interval,
                     'proj_method': self.custom_args.proj_method}
                ]
                if self.custom_args.optimizer_name == "Galore":
                    self.optimizer = GaLoreAdamW(param_groups, lr=self.args.learning_rate)
                    print("Using GaLoreAdamW optimizer")
                elif self.custom_args.optimizer_name == "HMT":
                    self.optimizer = HMTAdamW(param_groups, lr=self.args.learning_rate)
                    print("Using HMTAdamW optimizer")
                elif self.custom_args.optimizer_name == "Apollo":
                    self.optimizer = ApolloAdamW(param_groups, lr=self.args.learning_rate)
                    print("Using ApolloAdamW optimizer")
                elif self.custom_args.optimizer_name == "Lotus":
                    self.optimizer = Lotus(param_groups, lr=self.args.learning_rate)
                    print("Using Lotus optimizer")
                else:
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
            return self.optimizer


def main():
    args = parse_args()
    logger.info("Training parameters: %s", args)

    # 加载数据：若未传入 train_file/validation_file，则自动加载 SQuAD 数据集
    if args.train_file is not None and args.validation_file is not None:
        data_files = {"train": args.train_file, "validation": args.validation_file}
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset("squad")

    # 加载 tokenizer 和模型（针对问答任务）
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)

    # 预处理数据（训练与验证均采用滑动窗口分词）
    train_dataset = raw_datasets["train"]
    validation_dataset = raw_datasets["validation"]

    train_dataset = train_dataset.map(
        lambda examples: prepare_train_features(examples, tokenizer, args),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        lambda examples: prepare_train_features(examples, tokenizer, args),
        batched=True,
        remove_columns=validation_dataset.column_names,
    )
    # 保存原始验证数据，用于后处理答案
    eval_examples = raw_datasets["validation"]

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_total_limit=2,
    )

    # 构造自定义 Trainer（传入 custom_args 与原始 eval_examples）
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        custom_args=args,
        eval_examples=eval_examples,
    )

    # 开始训练
    trainer.train()

    # 预测与后处理，计算 SQuAD 指标
    logger.info("*** Evaluate ***")
    predictions = trainer.predict(validation_dataset)
    final_metrics = post_process_qa_predictions(
        eval_examples, validation_dataset, predictions.predictions, tokenizer, args
    )
    logger.info("Final metrics: %s", final_metrics)

    # 保存最终模型和 tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.enable_galore:
            optimizer = trainer.optimizer
            if hasattr(optimizer, 'param_groups'):
                for group in optimizer.param_groups:
                    if 'projector' in group:
                        projector = group['projector']
                        if projector is not None:
                            logger.info(f"[End of Training] => total subspace updates = {projector.update_count}")
                            if hasattr(projector, 'information'):
                                logger.info('\n'.join(map(str, projector.information)))
                        else:
                            logger.info("No projector found in this parameter group")
            
            if args.use_stable:
                logger.info("Stable subspace update was used")
            else:
                logger.info("Standard subspace update was used")

            if args.proj_method == "traditional":
                logger.info("Traditional projection method was used")
            else:
                logger.info("Power iteration projection method was used")


if __name__ == "__main__":
    import sys
    import wandb
    args = parse_args()

    wandb.init(
        project="squad-finetuning",  # 项目名称
        name=f"{args.model_name_or_path}-{args.optimizer_name}",  # 实验名称
        tags=["qa", "bert", args.optimizer_name],  # 标签，方便分类
        notes=f"Fine-tuning {args.model_name_or_path} on SQuAD with {args.optimizer_name}",  # 备注
        config=args,  # 记录所有超参数
    )

    class LoggerWriter:
        def __init__(self, level):
            self.level = level

        def write(self, message):
            if message.strip():
                self.level(message)

        def flush(self): 
            pass
    
    if args.optimizer_name == "Galore":
        method = "ft"
    elif args.optimizer_name == "HMT":
        method = "hmt"
    elif args.optimizer_name == "Apollo":
        method = "apollo"
    elif args.optimizer_name == "Lotus":
        method = "lotus"

    log_file_path = f'logfiles/{method}/{args.model_name_or_path}/squad_{args.lora_r}.log'

    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sys.stdout = LoggerWriter(logger.info)

    main()
