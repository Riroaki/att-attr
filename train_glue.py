from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import torch
import logging
import random
import numpy as np
from tqdm import tqdm, trange
from argparse import ArgumentParser
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from data_process import processors, num_labels_task, convert_examples_to_features, convert_features_to_dataset

# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt_{}.json'.format(args.task_name)), 'w'), sort_keys=True, indent=2)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

    amp_handle = None
    if args.fp16:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)

    # Prepare model
    if (args.model_recover_path is None) or len(args.model_recover_path) == 0:
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model, num_labels=num_labels)
    else:
        if not os.path.exists(args.model_recover_path):
            logger.info("Path does not exist: {0}".format(
                args.model_recover_path))
            sys.exit(0)
        logger.info(
            "***** Recover model: {0} *****".format(args.model_recover_path))
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model, state_dict=torch.load(args.model_recover_path), num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    if args.do_train:
        t_total = int(len(train_examples) / args.train_batch_size /
                      args.gradient_accumulation_steps * args.num_train_epochs)
    else:
        t_total = 1
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      correct_bias=False)
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.task_name == 'sts-b':
        if args.fp16:
            lbl_type = torch.half
        else:
            lbl_type = torch.float
    else:
        lbl_type = torch.long

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        train_data = convert_features_to_dataset(train_features, lbl_type)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        best_result = 0.0

        for i_epoch in trange(1, args.num_train_epochs+1, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                labels=label_ids)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    if amp_handle:
                        amp_handle._clear_cache()
                else:
                    loss.backward()

                tr_loss += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform validation
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
            eval_data = convert_features_to_dataset(eval_features, lbl_type)
            eval_segment = processor.get_dev_segments()[0]
            logger.info(
                "***** Running evaluation: {0}-{1} *****".format(eval_segment, i_epoch))
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_result = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            all_logits, all_label_ids = [], []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,
                                    labels=label_ids)
                    tmp_eval_loss = outputs[0]
                    logits = outputs[1]
                    if amp_handle:
                        amp_handle._clear_cache()

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                all_logits.append(logits)
                all_label_ids.append(label_ids)

                eval_loss += tmp_eval_loss.mean().item()

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            # compute evaluation metric
            all_logits = np.concatenate(all_logits, axis=0)
            all_label_ids = np.concatenate(all_label_ids, axis=0)
            metric_func = processor.get_metric_func()
            eval_result = metric_func(all_logits, all_label_ids)
            # logging the results
            logger.info(
                "***** Eval results for {0}: {1} *****".format(eval_segment, eval_result))
            if eval_result > best_result:
                best_result = eval_result
                # Save a trained model
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(
                    args.output_dir, "{0}.pt".format(args.task_name))
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("  Saved best model to {0}".format(
                    output_model_file))

    # delete unused variables
    del optimizer
    del param_optimizer
    del optimizer_grouped_parameters

    # Load a trained model that you have fine-tuned
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        del model

        output_model_file = os.path.join(
            args.output_dir, "{0}.pt".format(args.task_name))
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_set_list = []
        for eval_segment in processor.get_dev_segments():
            eval_examples = processor.get_dev_examples(
                args.data_dir, segment=eval_segment)
            eval_set_list.append((eval_segment, eval_examples))
            break

        for eval_segment, eval_examples in eval_set_list:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
            eval_data = convert_features_to_dataset(
                eval_features, lbl_type)
            logger.info("***** Running evaluation: %s *****", eval_segment)
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_result = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            all_logits, all_label_ids = [], []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,
                                    labels=label_ids)
                    tmp_eval_loss = outputs[0]
                    logits = outputs[1]
                    if amp_handle:
                        amp_handle._clear_cache()

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                all_logits.append(logits)
                all_label_ids.append(label_ids)

                eval_loss += tmp_eval_loss.mean().item()

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            # compute evaluation metric
            all_logits = np.concatenate(all_logits, axis=0)
            all_label_ids = np.concatenate(all_label_ids, axis=0)
            metric_func = processor.get_metric_func()
            eval_result = metric_func(all_logits, all_label_ids)
            # logging the results
            logger.info(
                "***** Eval results for {0}: {1} *****".format(eval_segment, eval_result))


if __name__ == "__main__":
    parser = ArgumentParser()

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
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    # Other parameters
    parser.add_argument("--bert_model",
                        default="./bert-base-uncased/",
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="./ckpt",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
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
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=8,
                        type=int,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
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
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    main(parser.parse_args())
