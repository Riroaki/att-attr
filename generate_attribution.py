from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse
import random
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from data_process import processors, num_labels_task, convert_examples_to_features

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale
        res = torch.cat([torch.add(baseline.unsqueeze(0), step*i)
                         for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i)
                         for i in range(num_points)], dim=0)
        return res, step_new[0]


def main(args):
    args.zero_baseline = True

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    if args.task_name == 'sts-b':
        lbl_type = torch.float
    else:
        lbl_type = torch.long

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # Load a trained model that you have fine-tuned
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Load dataset
    eval_segment = "dev_matched" if args.task_name == "mnli" else "dev"
    eval_examples = processor.get_dev_examples(
        args.data_dir, segment=eval_segment)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    # Load a fine-tuned model
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)
    model.eval()

    if args.bert_model.find("base") != -1:
        num_layers = 12
    elif args.bert_model.find("large") != -1:
        num_layers = 24

    # Prepare files to store attention scores & attribution scores
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)

    for index in tqdm(range(100)):
        eval_feature = eval_features[index]
        input_ids = torch.tensor([eval_feature.input_ids],
                                 dtype=torch.long).to(device)
        input_mask = torch.tensor(
            [eval_feature.input_mask], dtype=torch.long).to(device)
        segment_ids = torch.tensor(
            [eval_feature.segment_ids], dtype=torch.long).to(device)
        label_ids = torch.tensor([eval_feature.label_id],
                                 dtype=torch.long).to(device)

        # Attention scores & attribution scores' storage in each layer
        att_all, attr_all = [], []

        # Start from each layer
        for layer_index in range(num_layers):
            # Temporary storage for each forward-backward pass
            grad_tmp = []
            # Collect attention / gradients scores for integration
            for step in range(1, args.num_steps + 1):
                alpha = step / args.num_steps
                alpha_hook = model.bert.encoder.layer[layer_index].attention.self.register_forward_hook(
                    lambda module, input_data, output_data:
                    # Content of output_data:
                    # (self attention output, attention map)
                    # ([batch_size, seq_len, hidden_dim], [batch_size, num_heads, seq_len, seq_len])

                    # The scaled attention map won't appear in the later processes,
                    # so it's useless to do scaling on attention map.
                    # We perform scaling on attention output, because it is equivalent with scaling on attention map!
                    # Multiply it by alpha to get interpolated attention output
                    # Keep the attention map unchanged
                    (output_data[0] * alpha, output_data[1])
                )
                outputs = model(input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                labels=label_ids,
                                output_attentions=True)
                loss, attention = outputs[0], outputs[2]

                if alpha == 1:
                    logits = outputs[1][0]
                    pred_label = logits.argmax().item()
                    confidence = torch.softmax(logits, dim=-1).max().item()
                    logits = logits.detach().cpu().numpy()

                # To keep track of gradients of intermediate variables
                attention[layer_index].retain_grad()
                loss.backward()

                # Record gradient scores along original attention scores:
                # (batch_size, num_heads, seq_len, seq_len)
                grad_tmp.append(attention[layer_index].grad)

                # Remove hook and register another in next pass
                alpha_hook.remove()

            # Calculate attribution scores:
            # Attr = Ah / m * sum(grad_Ah) = Ah * mean(grad_Ah)
            att_scores = attention[layer_index]
            mean_grad = torch.mean(torch.stack(grad_tmp), dim=0)
            att_all.append(att_scores.squeeze(0).detach().cpu().numpy())
            attr_all.append(
                (att_scores * mean_grad).squeeze(0).detach().cpu().numpy())

        # Dump scores
        sample_data = {'input_ids': eval_feature.input_ids,
                       'attention': att_all,
                       'attribution': attr_all,
                       'logits': logits,
                       'prediction': pred_label,
                       'confidence': confidence,
                       'label': eval_feature.label_id}
        with open(os.path.join(args.output_dir, args.task_name,
                               '{0}-{1}.pkl'.format(eval_segment, index)), 'wb') as f:
            pickle.dump(sample_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The model file which will be evaluated.")

    # Other parameters
    parser.add_argument("--bert_model",
                        default="./bert-base-uncased/",
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="./output/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # # parameters about attention attribution
    parser.add_argument("--num_steps",
                        default=20,
                        type=int,
                        help="Num batch of an example.")

    args = parser.parse_args()
    main(parser.parse_args())
