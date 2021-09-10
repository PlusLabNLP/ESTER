# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer
from utils import *
from models import RobertaSpanPredictor
from optimization import *
from pathlib import Path
from collections import Counter
import re
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The load model directory")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=320, # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # construct model
    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        model = RobertaSpanPredictor.from_pretrained(args.model, mlp_hid=args.mlp_hid_size)
        if args.model_dir:
            model_state_dict = torch.load(args.model_dir + "/pytorch_model.bin")
            model.load_state_dict(model_state_dict)

    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    for split in ['dev']:
        eval_data = load_data(args.data_dir, "final_%s" % split, args.file_suffix)
        eval_features = convert_to_features(eval_data, tokenizer, max_length=args.max_seq_length, evaluation=True)

        eval_inputs = select_field(eval_features, 'inputs')

        eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
        eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)

        eval_offsets = select_field(eval_features, 'offsets')
        eval_labels = select_field(eval_features, 'labels')
        eval_events = select_field(eval_features, 'events')
        eval_answers = select_field(eval_features, 'answers')

        eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

        # flatten question_ids
        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        pred_answers = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, instance_indices = batch

            with torch.no_grad():
                eval_logits = model(input_ids, attention_mask=input_masks, token_type_ids=segment_ids)

                indices = instance_indices.cpu().tolist()
                offsets = [eval_offsets[i] for i in indices]

                eval_logits = filter_outputs(eval_logits, offsets)

                preds = unflatten_vector(torch.argmax(eval_logits, dim=-1), offsets)

                batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.tolist()]
                pred_answers.extend(get_answers(preds, offsets, batch_tokens))

        assert len(pred_answers) == len(eval_answers) == len(eval_events)
        eval_ems, F1, recl, prec, F1_e, r_e, pr_e, F1_a, r_a, pr_a, hit1 = [], [], [], [], [], [], [], [], [], [], []
        ps, rs, fs = {str(k): [] for k in range(1, 5)}, {str(k): [] for k in range(1, 5)}, {str(k): [] for k in range(1, 5)}

        for i, (pred, gold, event) in enumerate(zip(pred_answers, eval_answers, eval_events)):

            em = 1.0 if all([p in gold for p in pred]) and all([g in pred for g in gold]) else 0.0
            eval_ems.append(em)

            # construct unigram counter
            all_pred_counter = Counter()
            all_gold_counter = Counter()

            for n in range(1, 5):
                pred_counter = get_ngrams(pred, n=n)
                gold_counter = get_ngrams(gold, n=n)
                r, p, f = compute_f1(pred_counter, gold_counter)
                ps[str(n)].append(p)
                rs[str(n)].append(r)
                fs[str(n)].append(f)
                for k, v in pred_counter.items():
                    all_pred_counter[k] += v
                for k, v in gold_counter.items():
                    all_gold_counter[k] += v

            rl, pr, f1 = compute_f1(all_pred_counter, all_gold_counter)
            F1.append(f1)
            recl.append(rl)
            prec.append(pr)

            pred_counter = Counter([re.sub(r'[^\w\s]', '', s) for s in pred])
            gold_counter = Counter([re.sub(r'[^\w\s]', '', s) for s in gold])
            rl, pr, f1 = compute_f1(pred_counter, gold_counter)
            F1_a.append(f1)
            r_a.append(rl)
            pr_a.append(pr)

            rl, pr, f1 = compute_event_f1(pred, event)
            F1_e.append(f1)
            r_e.append(rl)
            pr_e.append(pr)

            if pred:
                hit1.append(compute_hit1(pred[0], event))
            else:
                hit1.append(0.0)

        logger.info("EM is %.4f" % np.mean(eval_ems))
        logger.info("Event HIT@1 is %.4f" % np.mean(hit1))
        logger.info("Overall Token Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                    (np.mean(recl), np.mean(prec), np.mean(F1)))
        logger.info("Answer Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                    (np.mean(r_a), np.mean(pr_a), np.mean(F1_a)))
        logger.info("Event Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                    (np.mean(r_e), np.mean(pr_e), np.mean(F1_e)))

        for n in range(1, 5):
            logger.info("%s-gram Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                        (n, np.mean(rs[str(n)]), np.mean(ps[str(n)]), np.mean(fs[str(n)])))

if __name__ == "__main__":
    main()
