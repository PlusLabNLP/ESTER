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
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path
import re
from collections import Counter

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
    parser.add_argument("--model_dir",
                        type=str,
                        help="saved model dir",
                        default="")
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
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
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
    if args.model_dir:
        logger.info(args.model_dir)
        model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")
        tokenizer = AutoTokenizer.from_pretrained(args.model, state_dict=model_state_dict)
        if 't5' in args.model:
            model = T5ForConditionalGeneration.from_pretrained(args.model, state_dict=model_state_dict)
        if 'bart' in args.model:
            model = BartForConditionalGeneration.from_pretrained(args.model, state_dict=model_state_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = T5ForConditionalGeneration.from_pretrained(args.model)

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
                             t_total=1)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split in ['dev']:
        eval_data = load_data(args.data_dir, "final_%s" % split, args.file_suffix)
        eval_features = convert_to_features_ans_gen(eval_data, ";", eval=True)

        eval_inputs = select_field(eval_features, 'inputs')
        eval_encoded_inputs = tokenizer(eval_inputs, padding=True, truncation=True, return_tensors="pt")

        eval_input_ids = eval_encoded_inputs['input_ids']
        eval_input_mask = eval_encoded_inputs['attention_mask']

        eval_key_indices = torch.tensor(list(range(len(eval_inputs))), dtype=torch.long)
        eval_types = select_field(eval_features, 'types')

        eval_labels = select_field(eval_features, 'labels')
        eval_encoded_outputs = tokenizer(eval_labels, padding=True, truncation=True, return_tensors="pt")
        eval_output_ids = eval_encoded_outputs['input_ids']

        logger.info("id_size: {}, mask_size: {}, instance_key_size: {}, label_size: {}".format(
            eval_input_ids.size(), eval_input_mask.size(), eval_key_indices.size(), eval_output_ids.size()))

        eval_events = select_field(eval_features, 'events')
        data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(data)
        eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        preds, golds, events, perplexity = [], [], [], []
        model.eval()

        type_indicators = {'Causal': [], 'Indicative Conditional': [], 'Sub-event': [],
                           'Counterfactual Conditional': [], 'Coreference': []}

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_masks, instance_indices, output_ids = batch

            with torch.no_grad():
                res = model.generate(input_ids, attention_mask=input_masks, max_length=128)
                preds.extend([x.split(";") for x in tokenizer.batch_decode(res, skip_special_tokens=True)])
                golds.extend([eval_labels[x].split(";") for x in instance_indices.tolist()])
                events.extend([eval_events[x] for x in instance_indices.tolist()])

                for i in instance_indices:
                    for tk in type_indicators:
                        if eval_types[i] in tk:
                            type_indicators[tk].append(1)
                        else:
                            type_indicators[tk].append(0)

        eval_em, eval_f1 = [], []

        ems, F1, recl, prec, F1_e, r_e, pr_e, hit1 = [], [], [], [], [], [], [], []
        for pred, gold, event in zip(preds, golds, events):
            logger.info(pred)
            logger.info(gold)
            logger.info(event)
            logger.info("="*50)
            em = 1.0 if all([p in gold for p in pred]) and all([g in pred for g in gold]) else 0.0
            eval_em.append(em)

            # construct unigram counter
            pred_counter = Counter([x for s in pred for x in re.sub(r'[^\w\s]', '', s).split(' ')])
            gold_counter = Counter([x for s in gold for x in re.sub(r'[^\w\s]', '', s).split(' ')])

            rl, pr, f1 = compute_unigram_f1(pred_counter, gold_counter)
            F1.append(f1)
            recl.append(rl)
            prec.append(pr)

            rl, pr, f1 = compute_event_f1(pred, event)
            F1_e.append(f1)
            r_e.append(rl)
            pr_e.append(pr)

            if f1 == 1.0:
                ems.append(1.0)
            else:
                ems.append(0.0)

            hit1.append(compute_hit1(pred[0], event))

        logger.info("Answer EM is %.4f" % np.mean(eval_em))
        logger.info("Event EM is %.4f" % np.mean(ems))
        logger.info("Event HIT@1 is %.4f" % np.mean(hit1))
        logger.info("Token Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                    (np.mean(recl), np.mean(prec), np.mean(F1)))
        logger.info("Event Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                    (np.mean(r_e), np.mean(pr_e), np.mean(F1_e)))

        logger.info("=" * 50)
        for key, idx in type_indicators.items():
            print("Eval %s questions" % key)
            assert len(idx) == len(golds)
            logger.info("Total %s questions (%.1f)" % (sum(idx), 100 * sum(idx) / len(golds)))

            ans_ems = [v for k, v in zip(idx, eval_em) if k == 1]
            temp_ems = [v for k, v in zip(idx, ems) if k == 1]

            temp_r = [v for k, v in zip(idx, recl) if k == 1]
            temp_pr = [v for k, v in zip(idx, prec) if k == 1]
            temp_f1s = [v for k, v in zip(idx, F1) if k == 1]

            temp_r_e = [v for k, v in zip(idx, r_e) if k == 1]
            temp_pr_e = [v for k, v in zip(idx, pr_e) if k == 1]
            temp_f1_e = [v for k, v in zip(idx, F1_e) if k == 1]

            temp_hit1 = [v for k, v in zip(idx, hit1) if k == 1]

            logger.info("Total %s QAs" % len(temp_ems))
            logger.info("Answer EM is %.4f" % np.mean(ans_ems))
            logger.info("Event EM is %.4f" % np.mean(temp_ems))
            logger.info("Event HIT@1 is %.4f" % np.mean(temp_hit1))
            logger.info("Token Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                  (np.mean(temp_r), np.mean(temp_pr), np.mean(temp_f1s)))
            logger.info("Event Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                  (np.mean(temp_r_e), np.mean(temp_pr_e), np.mean(temp_f1_e)))
            logger.info("=" * 20)

if __name__ == "__main__":
    main()
