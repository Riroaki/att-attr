from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset


def pred_argmax(out):
    return np.argmax(out, axis=1).reshape(-1)


def accuracy(out, labels):
    outputs = pred_argmax(out)
    r = accuracy_score(labels.reshape(-1), outputs)
    if np.isnan(r):
        r = 0.0
    return float(r)


def mcc(out, labels):
    outputs = pred_argmax(out)
    r = matthews_corrcoef(labels.reshape(-1), outputs)
    if np.isnan(r):
        r = 0.0
    return float(r)


def pearson_cc(out, labels):
    r = pearsonr(labels.reshape(-1), out.reshape(-1))[0]
    if np.isnan(r):
        r = 0.0
    return float(r)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, segment='train'):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, segment='dev'):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, segment='test'):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_train_segments(self):
        return ['train']

    def get_dev_segments(self):
        return ['dev']

    def get_test_segments(self):
        return ['test']

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_metric_func(self):
        return accuracy

    def get_pred(self, out):
        # default: classification
        lbl_list = self.get_labels()
        return [lbl_list[p] for p in pred_argmax(out).tolist()]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-2]
            text_b = line[-1]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev_matched'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, segment+".tsv")), segment)

    def get_test_examples(self, data_dir, segment='test_matched'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, segment+".tsv")), segment, is_test=True)

    def get_dev_segments(self):
        return ['dev_matched', 'dev_mismatched']

    def get_test_segments(self):
        return ['test_matched', 'test_mismatched', 'diagnostic']

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_metric_func(self):
        return mcc


class SstProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, segment='train'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if is_test:
                text_a = line[1]
                label = self.get_labels()[0]
            else:
                text_a = line[1]
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def _read_input_file(self, input_file):
        """Reads a tab separated value file."""
        lines = []
        with open(input_file, "r", encoding='utf-8') as f:
            for l in f:
                col_list = l.strip().split('\t')
                if len(col_list) == 6:
                    lines.append(col_list)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_input_file(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_input_file(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                if len(line) != 6:
                    print('Skip:', line)
                    continue
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            text_a = text_a.strip("\"")
            text_b = text_b.strip("\"")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = self.get_labels()[0]
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StsProcessor(DataProcessor):
    """Processor for the STS data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return None

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if is_test:
                text_a = line[-2]
                text_b = line[-1]
                label = str(random.random())
            else:
                text_a = line[-3]
                text_b = line[-2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_metric_func(self):
        return pearson_cc

    def get_pred(self, out):
        return [p for p in out.reshape(-1).tolist()]


class ScitailProcessor(DataProcessor):
    """Processor for the Scitail data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "scitail_1.0_train.tsv")), "train")

    def get_dev_examples(self, data_dir, segment='dev'):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "scitail_1.0_dev.tsv")), "dev")

    def get_test_examples(self, data_dir, segment='test'):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "scitail_1.0_test.tsv")), "test", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["neutral", "entails"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None

    features = []
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
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

        if label_map:
            label_id = label_map[example.label]
        else:
            label_id = float(example.label)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_features_to_dataset(features, lbl_type):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=lbl_type)
    all_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return all_data


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "sst-2": SstProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "wnli": WnliProcessor,
    "sts-b": StsProcessor,
    "scitail": ScitailProcessor,
}

num_labels_task = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "rte": 2,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "wnli": 2,
    "sts-b": 1,
    "scitail": 2,
}
