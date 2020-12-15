from __future__ import absolute_import, division, print_function
import csv
import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
import sklearn
from torch import Tensor
from eunjeon import Mecab
mecab = Mecab()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, row=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentence_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return pd.DataFrame(lines)

class MultiClassProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("Training {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "insu_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("Valid {}".format(os.path.join(data_dir, "valid.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "insu_valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("Test {}".format(os.path.join(data_dir, "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "insu_test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        labels =[]
        lines = self._read_tsv(os.path.join(data_dir, "labels.tsv"))
        for line in range(len(lines)):
            labels.append(str(line))
        return labels

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, row) in enumerate(df.values):
            guid = "%s-%s" % (set_type, i)
            label = row[0]
            text_a = row[1]
            # text_b = row[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MultiLabelProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("Training {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "labor_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("Valid {}".format(os.path.join(data_dir, "valid.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "labor_valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("Test {}".format(os.path.join(data_dir, "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "labor_test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        labels =[]
        lines = self._read_tsv(os.path.join(data_dir, "labels.tsv"))
        for line in range(len(lines)):
            labels.append(str(line))
        return labels

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, row) in enumerate(df.values):
            guid = "%s-%s" % (set_type, i)
            label = row[0]
            text_a = row[1]
            text_b = row[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


### kyoungman.bae @ 19-05-28 @
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, multi_label):
    """Loads a data file into a list of `InputBatch`s."""
    # label_map = {label: i for i, label in enumerate(label_list)}

    if multi_label==False :
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = do_lang(example.text_a)
        tokens_a = tokenizer.tokenize(tokens_a)

        tokens_b = None
        if example.text_b:
            ### kyoungman.bae @ 19-05-30
            tokens_b = do_lang(example.text_b)
            tokens_b = tokenizer.tokenize(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = []
        # for label in example.label:
        #     label_id.append(float(label))
        # label_id = label_map[example.label]

        if multi_label :
            label_id = [float(x) for x in example.label]
        else :
            label_id = label_map[example.label]


        if ex_index < 5 :
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("labels_id : {}".format(label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    true_label = labels
    return np.sum(outputs == true_label)

def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    # if sigmoid: y_pred = y_pred.sigmoid()
    if sigmoid: y_pred = torch.sigmoid(y_pred)
    #     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()

def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case

    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)

    return [np.mean(acc_list), sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None), sklearn.metrics.hamming_loss(y_true, y_pred)]

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def do_lang(text):
    mecab_pos = mecab.pos(text)
    return_result = ''
    for i in mecab_pos:
        temp_text = i[0] + '/' + i[1] + ' '
        return_result = return_result + temp_text

    return return_result