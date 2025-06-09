'''
entry point of prunePEFT
'''

from src.frequency_cnt import fre_cnt, transform_prune_dict
from pruning_methods_classed import PruneModel
from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
import torch
from utils.save_submit import saveInstance
from src.frequency_cnt import parse_log_files

import yaml
import argparse
import logging
import time
from copy import deepcopy
import os
import json
import random
import numpy as np


def print_trainable_parameters(model, logger):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


logger = logging.getLogger('controller')
logger.setLevel(logging.INFO) 
time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
day_str = time.strftime('%Y-%m-%d', time.localtime())
output_dir = f'outputs/{day_str}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_handler = logging.FileHandler(
    f'outputs/{day_str}/output_{time_str}.log', mode='w')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--result', type=str, default='result.json')


def comma_separated_strings(value):
    return value.split(',')


parser.add_argument('--device', type=comma_separated_strings, default=[0])
args = parser.parse_args()

args.device = [int(i) for i in args.device]

with open(args.method, 'r') as file:
    method_configs = yaml.safe_load(file)

with open(args.task, 'r') as file:
    task_configs = yaml.safe_load(file)

logger.info(
    f'Start exp for {args.task}:{task_configs}\n{args.method}:{method_configs}')

print(method_configs)
print(task_configs)

# hyperparameterization
if 'ADAPTER' in method_configs:
    logger.info(
        f"PRUNE_TURN: {method_configs['PRUNE_TURN']}, EPOCHS: {method_configs['EPOCHS']}, lora_lr: {method_configs['LORA_LR']}, adapter_lr: {method_configs['ADAPTER_LR']}, head_lr = 2e-4, warmup = 0.01, Batch = 32, adamw, betas=(0, 0.999), num_cycles =0.5, linear, turn = 4"
    )
else:
    logger.info(
        f"PRUNE_TURN: {method_configs['PRUNE_TURN']}, EPOCHS: {method_configs['EPOCHS']}, lora_lr: {method_configs['LORA_LR']}, head_lr = 2e-4, warmup = 0.01, Batch = 16, adamw, beta = 0, num_cycles = 1"
    )


def reset_seed(num=42):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num)
        torch.cuda.manual_seed_all(num) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if 'LORA' in method_configs:
    peft_type = 'LORA'
elif 'ADAPTER' in method_configs:
    peft_type = 'ADAPTER'
elif 'DORA' in method_configs:
    peft_type = 'DORA'
print(peft_type)

is_half = False

# run for each dataset
for ds_meta in task_configs['DATASETS']:
    dataset_name = ds_meta['DATASET_NAME']
    task_name = ds_meta['TASK_NAME']
    logger.info(f"dataset: {ds_meta}")

    if 'METRICS' in ds_meta:
        metric_list = ds_meta['METRICS']
    else:
        metric_list = ['accuracy']
    if 'SUBMIT' in task_configs:
        submit_flag = task_configs['SUBMIT']
    else:
        submit_flag = False

    configs = deepcopy(method_configs)
    configs['LOSS'] = ds_meta['LOSS']

    dataset = PEFTDataset(
        dataset_name,
        task_name,
        train_size=task_configs['TRAIN_SIZE'],
        test_size=task_configs['TEST_SIZE']).get_tokenized_dataset()

    prune_dataset = PEFTDataset(
        dataset_name,
        task_name,
        train_size=1.0,
        test_size=task_configs['TEST_SIZE'],
        shuffle=True).balanced_split(
            task_configs['PRUNE_SIZE']).get_tokenized_dataset()

    reset_seed()

    ####### warm-up #######
    #######################
    task_configs['WARM_UP'] = False
    if task_configs['WARM_UP']:
        warmup_dataset = PEFTDataset(
            dataset_name,
            task_name,
            train_size=task_configs['WARM_UP'],
            test_size=2,
            shuffle=True).get_tokenized_dataset()
        time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        day_str = time.strftime('%Y-%m-%d', time.localtime())

        warmup_logger = logging.getLogger('warmup')
        warmup_logger.setLevel(logging.INFO)
        output_dir = f'warmup_output/{day_str}_{time_str}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_handler2 = logging.FileHandler(
            f'{output_dir}/warmup_{task_name}_{time_str}.log', mode='a')
        file_handler2.setLevel(logging.INFO)
        formatter2 = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler2.setFormatter(formatter2)
        warmup_logger.addHandler(file_handler2)
        method_list = [
            'snip', 'activation', 'gradient', 'minimum_weight', 'zeros',
            'values_below_threshold'
        ]
        methods_with_blocks = {
            'lora_80': ['zeros'],
            'lora_50': ['values_below_threshold'],
            'adapter_80': ['minimum_weight', 'snip'],
            'lora_100': ['gradient'],
            'adapter_20': ['activation', 'minimum_weight', 'snip', 'zeros'],
            'lora_20': ['activation', 'values_below_threshold'],
            'adapter_50': ['snip', 'minimum_weight'],
            'adapter_100': ['snip']
        } # the original fixed blocks setting, if use warm-up, this will be replaced by the warm-up result
        pruner = PruneModel(methods_with_blocks)
        warmup_logger.info(f'WARMUP_ROUND: {method_configs["WARMUP_ROUND"]}')
        for i in range(method_configs['WARMUP_ROUND']):
            for prune_method in method_list:
                configs = deepcopy(method_configs)
                configs['LOSS'] = ds_meta['LOSS']

                warmup_logger.info(f'Prune method {prune_method}')
                for prune_turn in range(int(configs['WARMUP_TURN'])):

                    configs['EPOCHS'] = configs['WARMUP_EPOCHS']
                    model = None
                    gradients = None
                    activations = None
                    torch.cuda.empty_cache()

                    warmup_logger.info(
                        f'Start searching for LORA:{configs["LORA"]}; ADAPTER:{configs["ADAPTER"]}'
                    )

                    print(configs)
                    if is_half:
                        model = PEFTModel(configs, warmup_dataset).half()
                    else:
                        model = PEFTModel(configs, warmup_dataset)

                    res, gradients, activations, mid_submit_res = model.run(
                        args.device,
                        submit_flag=False,
                        val_flag=False,
                        use_original_trainer=False)
                    warmup_logger.info(
                        f'Mid-Result {res} for LORA:{configs["LORA"]}; ADAPTER:{configs["ADAPTER"]}'
                    )

                    idx_list = pruner.prune_model(
                        model.model,
                        task_name='my_module',
                        opts=['lora', 'adapter'],
                        p_method=prune_method,
                        top_p=12,
                        print_names=False,
                        gradients=gradients,
                        activations=activations)

                    for idx_dic in idx_list:
                        idx = idx_dic['layer_number']
                        idt = idx_dic['layer_type']

                        warmup_logger.info(f'Pruned layer: {idx, idt}')
                        configs[idt.upper()][int(idx)] = 0

                    model = None
                    gradients = None
                    activations = None
                    torch.cuda.empty_cache()
        stat = parse_log_files(output_dir)
        freq = fre_cnt(stat)
        methods_with_blocks = transform_prune_dict(freq)
        print(methods_with_blocks)
        warmup_logger.info(f'warmup res: {methods_with_blocks}')

        warmup_logger.removeHandler(file_handler2)
    else:
        methods_with_blocks = {
            'lora_80': ['zeros'],
            'lora_50': ['values_below_threshold'],
            'adapter_80': ['minimum_weight', 'snip'],
            'lora_100': ['gradient'],
            'adapter_20': ['activation', 'minimum_weight', 'snip', 'zeros'],
            'lora_20': ['activation', 'values_below_threshold'],
            'adapter_50': ['snip', 'minimum_weight'],
            'adapter_100': ['snip']
        }
    #######################

    # run base model(without prune)
    res_methods = {}
    if task_configs['RUN_BASE'] == True:
        logger.info(f'Base')
        if is_half:
            model = PEFTModel(configs, dataset).half()
        else:
            model = PEFTModel(configs, dataset)
        res, _, _, submit_res = model.run(
            args.device, submit_flag=submit_flag, use_original_trainer=False)
        logger.info(f'First-Result {res} for {configs[peft_type]}')
        model = None
        torch.cuda.empty_cache()

        score = 0.0
        for metric_name in metric_list:
            score += res[metric_name]
        score /= len(metric_list)
        res['score'] = score
        res_methods['base'] = res
        saver = saveInstance(dataset_name, task_name, 'base', day_str, time_str)
        saver.save_submit(submit_res)

    if not task_configs['PRUNE_METHODS']:
        task_configs['PRUNE_METHODS'] = []

    for prune_method in task_configs['PRUNE_METHODS']:
        logger.info(f'Prune method {prune_method}')
        configs = deepcopy(method_configs)
        configs['LOSS'] = ds_meta['LOSS']
        origin_epochs = configs['EPOCHS']
        pruner = PruneModel(methods_with_blocks)
        for prune_turn in range(int(configs['PRUNE_TURN'])):

            configs['EPOCHS'] = configs['PRUNE_EPOCHS']
            reset_seed()
            model = None
            gradients = None
            activations = None
            torch.cuda.empty_cache()

            logger.info(
                f'Start searching for LORA:{configs["LORA"]}; ADAPTER:{configs["ADAPTER"]}'
            )
            if is_half:
                model = PEFTModel(configs, prune_dataset).half()
            else:
                model = PEFTModel(configs, prune_dataset)
            res, gradients, activations, mid_submit_res = model.run(
                args.device, submit_flag=submit_flag, val_flag=False)
            logger.info(
                f'Mid-Result {res} for LORA:{configs["LORA"]}; ADAPTER:{configs["ADAPTER"]}'
            )

            idx_list = pruner.prune_model(
                model.model,
                task_name='my_module',
                opts=['lora', 'adapter'],
                p_method=prune_method,
                top_p=4,
                print_names=False,
                gradients=gradients,
                activations=activations)

            for idx_dic in idx_list:
                idx = idx_dic['layer_number']
                idt = idx_dic['layer_type']

                logger.info(f'Pruned layer: {idx, idt}')
                configs[idt.upper()][int(idx)] = 0

            model = None
            gradients = None
            activations = None
            torch.cuda.empty_cache()

        # retrain，save submit result
        if int(configs['PRUNE_TURN']) != 0:
            configs['EPOCHS'] = origin_epochs
            if is_half:
                model = PEFTModel(configs, dataset).half()
            else:
                model = PEFTModel(configs, dataset)
            print_trainable_parameters(model.model, logger)

            saver = saveInstance(dataset_name, task_name, prune_method, day_str,
                                 time_str)
            res, _, _, submit_res = model.run(
                args.device,
                submit_flag=submit_flag,
                saver=saver,
                use_original_trainer=False)
            logger.info(
                f'Final-Result {res} for LORA:{configs["LORA"]}; ADAPTER:{configs["ADAPTER"]}'
            )
            score = 0.0
            for metric_name in metric_list:
                score += res[metric_name]
            score /= len(metric_list)
            res['score'] = score
            res_methods[prune_method] = res
            saver = saveInstance(dataset_name, task_name, prune_method, day_str,
                                 time_str)
            saver.save_submit(submit_res)

