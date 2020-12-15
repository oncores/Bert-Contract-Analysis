"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, FINETUNED_NAME
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, \
    MultiClassification, MultiLabelClassification

### kyoungman.bae @ 19-05-28 
from pytorch_pretrained_bert.tokenization_morp import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from preprocess_function import accuracy, MultiClassProcessor, MultiLabelProcessor, convert_examples_to_features, hamming_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def rcml_main(args):
    logger.info('do-train : {}'.format(args.do_train))
    logger.info('do-valid : {}'.format(args.do_eval))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visua`lstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.multi_label :
        processor = MultiLabelProcessor()
    else :
        processor = MultiClassProcessor()

    converter = convert_examples_to_features

    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    train_sen_examples = None
    eval_sen_examples = None
    test_sen_examples = None
    num_train_optimization_steps = None

    if args.do_train:
        train_sen_examples = processor.get_train_examples(args.data_dir)
        eval_sen_examples = processor.get_dev_examples(args.data_dir)

        train_sen_features = converter(train_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)
        eval_sen_features = converter(eval_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)

        num_train_optimization_steps = int(
            len(train_sen_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.do_eval:
        eval_sen_examples = processor.get_dev_examples(args.data_dir)
        eval_sen_features = converter(eval_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)

    if args.do_test:
        test_sen_examples = processor.get_test_examples(args.data_dir)
        test_sen_features = converter(test_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    if args.do_train :
        if args.multi_label :
            model = MultiLabelClassification.from_pretrained(args.bert_model_path,
                                                             args.bert_model_name,
                                                             cache_dir=cache_dir,
                                                             num_labels=num_labels)
        else :
            model = MultiClassification.from_pretrained(args.bert_model_path,
                                                        args.bert_model_name,
                                                        cache_dir=cache_dir,
                                                        num_labels=num_labels)

    elif args.do_eval or args.do_test :
        model = torch.load(os.path.join(args.bert_model_path, args.bert_model_name))
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
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        if args.do_train == True:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    ##train_model
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        model.unfreeze_bert_encoder()

        if len(train_sen_features) == 0:
            logger.info("The number of train_features is zero. Please check the tokenization. ")
            sys.exit()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_sen_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_sen_input_ids = torch.tensor([f.input_ids for f in train_sen_features], dtype=torch.long)
        train_sen_input_mask = torch.tensor([f.input_mask for f in train_sen_features], dtype=torch.long)
        train_sen_segment_ids = torch.tensor([f.segment_ids for f in train_sen_features], dtype=torch.long)
        train_sen_label_ids = torch.tensor([f.label_id for f in train_sen_features], dtype=torch.long)

        eval_sen_input_ids = torch.tensor([f.input_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_input_mask = torch.tensor([f.input_mask for f in eval_sen_features], dtype=torch.long)
        eval_sen_segment_ids = torch.tensor([f.segment_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_label_ids = torch.tensor([f.label_id for f in eval_sen_features], dtype=torch.long)

        train_sen_data = TensorDataset(train_sen_input_ids, train_sen_input_mask, train_sen_segment_ids, train_sen_label_ids)
        eval_sen_data = TensorDataset(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_sen_label_ids)

        if args.local_rank == -1:
            train_sen_sampler = RandomSampler(train_sen_data)
            eval_sen_sampler = RandomSampler(eval_sen_data)

        else:
            train_sen_sampler = DistributedSampler(train_sen_data)
            eval_sen_sampler = DistributedSampler(eval_sen_data)

        train_sen_dataloader = DataLoader(train_sen_data, batch_size=args.train_batch_size, worker_init_fn=lambda _: np.random.seed())
        eval_sen_dataloader = DataLoader(eval_sen_data, batch_size=args.train_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()

            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, train_sen_batch in enumerate(tqdm(train_sen_dataloader, total=len(train_sen_dataloader), desc="Iteration")):

                train_sen_batch = tuple(t.to(device) for t in train_sen_batch)
                sen_input_ids, sen_input_mask, sen_segment_ids, sen_label_ids = train_sen_batch

                loss = model(sen_input_ids, sen_input_mask, sen_segment_ids, sen_label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * WarmupLinearSchedule(
                            global_step / num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predicted_labels, target_labels = list(), list()

            for eval_sen_batch in eval_sen_dataloader:
                eval_sen_batch = tuple(t.to(device) for t in eval_sen_batch)
                eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_label_ids = eval_sen_batch

                with torch.no_grad():
                    model.eval()
                    tmp_eval_loss = model(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_label_ids)
                    senten_encoder, temp_eval_logits = model(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids)

                # logits = temp_eval_logits.detach().cpu().numpy()
                eval_label_ids = eval_label_ids.to('cpu').numpy()

                if args.multi_label :
                    predicted_labels.extend(torch.sigmoid(temp_eval_logits).round().long().cpu().detach().numpy())
                    target_labels.extend(eval_label_ids)
                else :
                    argmax_logits = np.argmax(torch.softmax(temp_eval_logits, dim=1).detach().cpu().numpy(), axis=1)
                    predicted_labels.extend(argmax_logits)
                    target_labels.extend(eval_label_ids)

                    tmp_eval_accuracy = accuracy(temp_eval_logits.detach().cpu().numpy(), eval_label_ids)
                    eval_accuracy += tmp_eval_accuracy

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                nb_eval_examples += eval_sen_input_ids.size(0)

            epoch_ev_loss = eval_loss / nb_eval_steps
            epoch_tr_loss = tr_loss / nb_tr_steps

            target_labels = np.array(target_labels)
            predicted_labels = np.array(predicted_labels)

            if args.multi_label:
                eval_micro_metric = metrics.f1_score(target_labels, predicted_labels, average='micro')
                ham_score = hamming_score(target_labels, predicted_labels)
            else :
                eval_micro_metric = metrics.f1_score(target_labels, predicted_labels, average='micro')

            logger.info('')
            logger.info('################### epoch ################### : {}'.format(epoch + 1))
            logger.info('################### train loss ###################: {}'.format(epoch_tr_loss))
            logger.info('################### valid loss ###################: {}'.format(epoch_ev_loss))
            logger.info('################### valid micro f1 ###################: {}'.format(eval_micro_metric))

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), './model/eval_model/{}_epoch_{}_val_loss.bin'.format(epoch+1, epoch_ev_loss))
            torch.save(model, './model/eval_model/{}_epoch_{}_val_loss.pt'.format(epoch+1, epoch_ev_loss))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

        # torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        if args.multi_label:
            model = MultiLabelClassification(config, num_labels=num_labels)
        else:
            model = MultiClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_sen_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_sen_input_ids = torch.tensor([f.input_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_input_mask = torch.tensor([f.input_mask for f in eval_sen_features], dtype=torch.long)
        eval_sen_segment_ids = torch.tensor([f.segment_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_label_ids = torch.tensor([f.label_id for f in eval_sen_features], dtype=torch.long)

        eval_sen_data = TensorDataset(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_sen_label_ids)

        # Run prediction for full data
        eval_sen_sampler = RandomSampler(eval_sen_data)
        eval_sen_dataloader = DataLoader(eval_sen_data, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels, target_labels = list(), list()

        for eval_sen_batch in tqdm(eval_sen_dataloader, total=len(eval_sen_dataloader), desc="Evaluating"):
            eval_sen_batch = tuple(t.to(device) for t in eval_sen_batch)
            eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_label_ids = eval_sen_batch

            with torch.no_grad():
                tmp_eval_loss = model(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_label_ids)
                senten_encoder, temp_eval_logits = model(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids)

            label_ids = eval_label_ids.to('cpu').numpy()

            if args.multi_label :
                predicted_labels.extend(torch.sigmoid(temp_eval_logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids)
            else :
                argmax_logits = np.argmax(torch.softmax(temp_eval_logits, dim=1).cpu().detach().numpy(), axis=1)
                predicted_labels.extend(argmax_logits)
                target_labels.extend(eval_label_ids)

                tmp_eval_accuracy = accuracy(temp_eval_logits.detach().cpu().numpy(), label_ids)
                eval_accuracy += tmp_eval_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            nb_eval_examples += eval_sen_input_ids.size(0)

        eval_loss = eval_loss / nb_eval_steps

        if args.multi_label :
            eval_micro_metric = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')
        else :
            eval_metric = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')
        loss = tr_loss / nb_tr_steps if args.do_train else None
        logger.info('################### valid micro metric ###################: {}'.format(eval_micro_metric))

    if args.do_test:
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_sen_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_sen_input_ids = torch.tensor([f.input_ids for f in test_sen_features], dtype=torch.long)
        test_sen_input_mask = torch.tensor([f.input_mask for f in test_sen_features], dtype=torch.long)
        test_sen_segment_ids = torch.tensor([f.segment_ids for f in test_sen_features], dtype=torch.long)
        test_sen_label_ids = torch.tensor([f.label_id for f in test_sen_features], dtype=torch.long)

        test_sen_data = TensorDataset(test_sen_input_ids, test_sen_input_mask, test_sen_segment_ids, test_sen_label_ids)

        # Run prediction for full data
        test_sen_sampler = SequentialSampler(test_sen_data)
        test_sen_dataloader = DataLoader(test_sen_data, batch_size=args.eval_batch_size)

        all_logits = None
        all_labels = None
        all_encoders = None
        model.eval()
        test_accuracy = 0
        nb_test_examples = 0
        predicted_labels, target_labels = list(), list()

        for test_sen_batch in tqdm(test_sen_dataloader, total=len(test_sen_dataloader), desc='Prediction'):

            test_sen_batch = tuple(t.to(device) for t in test_sen_batch)
            test_sen_input_ids, test_sen_input_mask, test_sen_segment_ids, test_label_ids = test_sen_batch

            with torch.no_grad():
                sentence_encoder, logits = model(test_sen_input_ids, test_sen_input_mask, test_sen_segment_ids)

            label_ids = test_label_ids.to('cpu').numpy()

            if args.multi_label :
                predicted_labels.extend(torch.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids)
            else :
                argmax_logits = np.argmax(torch.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
                predicted_labels.extend(argmax_logits)
                target_labels.extend(label_ids)

            logits = logits.detach().cpu().numpy()
            sentence_encoder = sentence_encoder.detach().cpu().numpy()

            tmp_test_accuracy = accuracy(logits, label_ids)
            test_accuracy += tmp_test_accuracy
            nb_test_examples += test_label_ids.size(0)

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

            if all_labels is None:
                all_labels = label_ids
            else:
                all_labels = np.concatenate((all_labels, label_ids), axis=0)

            if all_encoders is None:
                all_encoders = sentence_encoder
            else:
                all_encoders = np.concatenate((all_encoders, sentence_encoder), axis=0)

            if args.multi_label:
                eval_micro_metric = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')
            else :
                eval_micro_metric = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')

        logger.info('################### test micro metric ###################: {}'.format(eval_micro_metric))

        input_data = [{'id': input_example.guid, 'text': input_example.text_a} for input_example in test_sen_examples]

        pred_logits = pd.DataFrame(all_logits, columns=label_list)
        pred_encoder = pd.DataFrame(all_encoders)
        pred_result = pd.concat([pred_logits, pred_encoder], axis=1)

        real_text = pd.DataFrame(input_data)
        real_label = pd.DataFrame(all_labels)
        real_result = pd.concat([real_text, real_label], axis=1)

        pred_result.to_csv('./output_dir/output_sentence_pred.csv', index=None)
        real_result.to_csv('./output_dir/output_sentence_real.csv', index=None)