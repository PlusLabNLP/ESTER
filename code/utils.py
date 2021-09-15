import json
import torch
import numpy as np
from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import logging
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import copy
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def cal_metrics(logits, golds, offsets, label_map, baseline=""):
    f1s, ems = [], []

    for b, (offset, gold) in enumerate(zip(offsets, golds)):
        pred = torch.argmax(logits[b, :, :], dim=1).detach().cpu().tolist()
        # filter out non-leading, question and pad tokens
        pred = [pred[x] for x in offset]

        if baseline == "random":
            pred = list(np.random.choice([0, 1], size=(len(pred),), p=[1./2, 1./2]))
        if baseline == "all-1s":
            pred = [1] * len(pred)
        if baseline == "all-0s":
            pred = [0] * len(pred)

        assert len(pred) == len(gold)
        f1 = cal_f1(pred, gold, label_map)
        f1s.append(f1)

        em = 1 if pred == gold else 0
        ems.append(em)

    return f1s, ems

def calculate_bleu(references, predidctions):
    smoothing = SmoothingFunction()
    bleu_scores = []
    for g, p in zip(references, predidctions):
        bleu_scores.append(sentence_bleu([g.split(' ')], p.split(' '), smoothing_function=smoothing.method1))
    return bleu_scores

def get_ngrams(alist, n=2):
    counter = Counter()
    for s in alist:
        cands = re.sub(r'[^\w\s]', '', s).split(' ')
        for i in range(len(cands)-n+1):
            key = ' '.join(cands[i:i+n])
            counter[key] += 1
    return counter

def compute_f1(pred_counter, gold_counter):
    corr = 0
    for w in pred_counter:
        if w in gold_counter:
            corr += min(pred_counter[w], gold_counter[w])

    prec = float(corr) / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0.0
    recl = float(corr) / sum(gold_counter.values()) if sum(gold_counter.values()) > 0 else 0.0
    return recl, prec, 2 * prec * recl / (prec + recl) if prec + recl > 0 else 0.0

def compute_unigram_f1(pred_counter, gold_counter):
    corr = 0
    for w in pred_counter:
        if w in gold_counter:
            corr += min(pred_counter[w], gold_counter[w])

    prec = float(corr) / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0.0
    recl = float(corr) / sum(gold_counter.values()) if sum(gold_counter.values()) > 0 else 0.0
    return recl, prec, 2 * prec * recl / (prec + recl) if prec + recl > 0 else 0.0

def compute_event_f1(pred_ans, events):
    if len(events) == 0:
        return 0.0, 0.0, 0.0
    c = [any([e in p for p in pred_ans]) for e in events]
    recl = sum(c) / len(events)

    c = [any([e in p for e in events]) for p in pred_ans]
    prec = sum(c) / len(pred_ans) if len(pred_ans) > 0 else 0.0

    return recl, prec, 2 * prec * recl / (prec + recl) if prec + recl > 0 else 0.0

def compute_hit1(pred_ans, events):

    if len(events) == 0 or not any([e in pred_ans for e in events]):
        return 0.0
    else:
        return 1.0

def map_span_to_event(pred, event):
    pred_event = [0] * len(event)
    match_pred_tok, in_span = -1, 0
    for i, (p, e) in enumerate(zip(pred, event)):
        if p == 1:
            if e == 1:
                match_pred_tok = i
                pred_event[i] = 1
            in_span = 1
        else:
            # spacial case: was in span, but no matched event tok
            if in_span and match_pred_tok == -1:
                pred_event[i-1] = 1
            match_pred_tok, in_span = -1, 0

    # termination corner case: last tok in span, but no matched event tok
    if in_span and match_pred_tok == -1:
        pred_event[i] = 1

    return pred_event

def cal_event_based_metrics(logits, events, offsets, label_map, baseline=""):
    f1s, ems = [], []

    for b, (offset, event) in enumerate(zip(offsets, events)):
        pred = torch.argmax(logits[b, :, :], dim=1).detach().cpu().tolist()
        # filter out non-leading, question and pad tokens
        pred = [pred[x] for x in offset]
        assert len(pred) == len(event)

        pred_event = map_span_to_event(pred, event)

        if baseline == "random":
            pred_event = list(np.random.choice([0, 1], size=(len(pred),), p=[1./2, 1./2]))
        if baseline == "all-1s":
            pred_event = [1] * len(pred)
        if baseline == "all-0s":
            pred_event = [0] * len(pred)

        assert len(pred_event) == len(event)
        f1 = cal_f1(pred_event, event, label_map)
        f1s.append(f1)
        em = 1 if pred_event == event else 0
        ems.append(em)

    return f1s, ems

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def cal_f1(pred_labels, true_labels, label_map, exclude_labels=[0], log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    # we only need positive f1 score
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count

    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score

def select_field(data, field):
    # collect a list of field in data
    # fields: 'label_start', 'label_end', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'
    return [ex[field] for ex in data]

def load_data(data_dir, split, suffix):
    filename = "%s%s%s" % (data_dir, split, suffix)
    print("==========load data from %s ===========" % filename)
    with open(filename, "r") as read_file:
        return json.load(read_file)

def filter_outputs(logits, offsets):
    filtered_logits = []
    for b, offset in enumerate(offsets):
        for i in range(len(offset)):
            filtered_logits.append(logits[b, offset[i], :].unsqueeze(0))

    return torch.cat(filtered_logits, dim=0)


def flatten_vector(vector):
    return [v for vec in vector for v in vec]

def unflatten_vector(vector, offsets):

    new_vectors = []
    start = 0
    for b in range(len(offsets)):
        offset = len(offsets[b])
        new_vectors.append(vector[start:start+offset])
        start += offset

    assert start == len(vector)
    return new_vectors

def replace_special(astr, is_leading=False):
    if is_leading:
        return astr.replace(chr(288), '').replace(chr(266), '')
    else:
        return astr.replace(chr(288), ' ').replace(chr(266), '\n')

def get_answers(preds, offsets, tokens):
    batch_answers = []
    for b in range(len(offsets)):
        pred, offset = preds[b], offsets[b]
        assert len(pred) == len(offset)
        answers = []
        in_answer, cur_ans = False, ""
        for p, o in zip(pred, offset):
            if p == 0 and in_answer:
                answers.append(cur_ans)
                in_answer, cur_ans = False, ""
            if p == 1:
                is_leading_token = False if in_answer else True
                cur_ans += replace_special(tokens[b][o], is_leading=is_leading_token)
                in_answer = True
            if p == 2:
                if in_answer:
                    answers.append(cur_ans)
                    cur_ans = ""
                cur_ans += replace_special(tokens[b][o], is_leading=True)
                in_answer = True

        if cur_ans:
            answers.append(cur_ans)
        batch_answers.append(answers)
    return batch_answers


def convert_to_features(data, tokenizer, max_length=300, evaluation=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    max_len_global = 0  # to show global max_len without truncating
    mismatch = 0
    for v in data:

        question = tokenizer.tokenize(v['question'].lower())
        context = tokenizer.tokenize(v['context'].lower())
        answers = [ans.lower() for ans in v['answer_texts']]
        ans_starts = [int(ans.split(',')[0][1:]) for ans in v['answer_indices']]
        ans_ends = [int(ans.split(',')[1][:-1]) for ans in v['answer_indices']]
        answers_to_match = ["" for _ in range(len(answers))]

        offset = 0

        labels = []
        for token in context:
            temp = copy.copy(token)
            is_answer = 0.0
            for i, (s, e) in enumerate(zip(ans_starts, ans_ends)):
                if s <= offset + len(token) <= e:
                    # dealing leading special tokens
                    if not answers_to_match[i]:
                        if token[:3] == chr(288) + "``":
                            token = token[3:]
                        elif token[:2] in [chr(288) + '"', chr(288) + '``', chr(288) + '(']:
                            token = token[2:]
                        elif token[0] in [chr(266), chr(288), '"', '(', '``']:
                            token = token[1:]
                    else:
                        if token[0] == chr(288):
                            token = token.replace(chr(288), ' ')
                    if token:
                        is_answer = 1.0 if answers_to_match[i] else 2.0
                    answers_to_match[i] += token
            offset += len(temp)
            labels.append(is_answer)

        assert len(labels) == len(context)

        # answer mismatch is very rare, usefully caused by some special tokens. Minor impact and skip for now.
        for a, b in zip(answers_to_match, answers):
            if a != b:
                mismatch += 1

        bos, sep, eos = tokenizer.bos_token, tokenizer.sep_token, tokenizer.eos_token
        # two sep used in RoBERTa
        tokenized_ids = tokenizer.convert_tokens_to_ids([bos] + question + [sep]*2 + context + [eos])
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # bos + 2*sep + all question tokens
        offsets = [len(question) + 3 + k for k in range(len(labels))]

        sample = {'labels': labels,
                  'types': v['type'],
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'offsets': offsets}
        if evaluation:
            sample['answers'] = answers
            sample['events'] = [x.lower() for x in v['events']]
            sample['inputs'] = v['context']

        # check some example data
        if len(samples) < 00:
            logger.info(sample)
            for t, l in zip(context, labels):
                if l == 1.0:
                    logger.info("%s ; %s" % (t, l))
            logger.info("===========")
        samples.append(sample)
    logger.info(mismatch)
    logger.info("Maximum length after tokenization is: % s" % (max_len_global))
    return samples


class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0
        self.res_dict = {}

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]


        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            total_count = self.total_truths.get(label, 0)
            pred_count = self.total_predictions.get(label, 0)

            n_true += total_count
            n_pred += pred_count

            correct_count = self.confusion_mat[label][label]
            n_correct += correct_count

            precision = safe_division(correct_count, pred_count)
            recall = safe_division(correct_count, total_count)
            f1_score = safe_division(2 * precision * recall, precision + recall)
            all_scores.append((precision, recall, f1_score))
            self.res_dict[label] = (f1_score, total_count)
            row = [total_count, pred_count, correct_count, precision, recall, f1_score]
            row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
            row.insert(0, display_label.rjust(max_label_width))
            res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res


def convert_to_features_ans_gen(data, sep_tok, eval=False, leaderboard=False):
    samples = []
    counter = 0

    for v in data:
        inputs = "%s \\n %s" % (v['question'].lower(), v['context'].lower())

        if eval:
            sample = {'labels': sep_tok.join([x.lower() for x in v['answer_texts']]),
                      'events': [x.lower() for x in v['events']],
                      'types': v['type'],
                      'inputs': inputs}
        elif leaderboard:
            sample = {'inputs': inputs}
        else:
            sample = {'labels': sep_tok.join([x.lower() for x in v['answer_texts']]),
                      'types': v['type'],
                      'inputs': inputs}

        counter += 1
        # check some example data
        if counter < 1:
            print(sample)
        samples.append(sample)
    return samples


