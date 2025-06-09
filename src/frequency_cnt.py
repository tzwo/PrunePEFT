import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_log_files(log_folder):
    '''
    read log files and parse the prune information
    '''
    prune_methods_stats = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(log_folder):
        for log_file in files:
            if log_file.endswith(".log"):
                with open(os.path.join(root, log_file), 'r') as file:
                    current_prune_method = None
                    for line in file:
                        method_match = re.search(r"Prune method (\w+)", line)
                        if method_match:
                            turn = 0
                            current_prune_method = method_match.group(1)

                        layer_match = re.search(
                            r"Pruned layer: \('(\d+)', '(\w+)'\)", line)
                        if layer_match and current_prune_method:
                            turn += 1
                            layer_id = int(
                                layer_match.group(1))  
                            method = layer_match.group(2)

                            if 1 <= turn <= 5:
                                prune_methods_stats[
                                    f'{current_prune_method}_1-5'][
                                        f'{layer_id}:{method}'] += 1
                            elif 6 <= turn <= 10:
                                prune_methods_stats[
                                    f'{current_prune_method}_6-10'][
                                        f'{layer_id}:{method}'] += 1
                            elif 11 <= turn <= 15:
                                prune_methods_stats[
                                    f'{current_prune_method}_11-15'][
                                        f'{layer_id}:{method}'] += 1
                            prune_methods_stats[f'{current_prune_method}'][
                                f'{layer_id}:{method}'] += 1
                            prune_methods_stats[f'{current_prune_method}'][
                                f'{layer_id}'] += 1

    return prune_methods_stats


def parse_log_file(log_file_path):
    prune_methods_stats = defaultdict(lambda: defaultdict(int))

    with open(log_file_path, 'r') as file:
        current_prune_method = None
        for line in file:
            # prune method
            method_match = re.search(r"Prune method (\w+)", line)
            if method_match:
                current_prune_method = method_match.group(1)

            # the layer and module that are pruned
            layer_match = re.search(r"Pruned layer: \('(\d+)', '(\w+)'\)", line)
            if layer_match and current_prune_method:
                turn = 0  
                layer_id = int(layer_match.group(1))  # for sort
                method = layer_match.group(2)

                if 1 <= turn <= 5:
                    prune_methods_stats[f'{current_prune_method}_1-5'][
                        f'{layer_id}:{method}'] += 1
                elif 6 <= turn <= 10:
                    prune_methods_stats[f'{current_prune_method}_6-10'][
                        f'{layer_id}:{method}'] += 1
                elif 11 <= turn <= 15:
                    prune_methods_stats[f'{current_prune_method}_11-15'][
                        f'{layer_id}:{method}'] += 1
                prune_methods_stats[f'{current_prune_method}'][
                    f'{layer_id}:{method}'] += 1
                prune_methods_stats[f'{current_prune_method}'][
                    f'{layer_id}'] += 1

    return prune_methods_stats


def fre_cnt(stats):
    '''count the frequency of different pruning methods'''
    frequency = {}
    for method, layer_stats in stats.items():
        if '-' in method:
            continue
        # sort by layer id
        frequency[method] = {}
        sorted_layers = sorted(
            layer_stats.keys(),
            key=lambda x: (int(x.split(':')[0]), ':' not in x, x))

        frequencies = [layer_stats[layer] for layer in sorted_layers]

        # calculate the total number of LORA and ADAPTER
        lora_count = sum(
            freq for layer, freq in layer_stats.items() if 'LORA' in layer)
        adapter_count = sum(
            freq for layer, freq in layer_stats.items() if 'ADAPTER' in layer)

        # count the number of LORA and ADAPTER in different percentage ranges
        total_layers = max(
            int(layer.split(':')[0]) for layer in layer_stats.keys()) + 1

        frequency[method]['pct20_count_lora'] = sum(
            freq for layer, freq in layer_stats.items()
            if int(layer.split(':')[0]) < total_layers * 0.2 and ":" in layer and 'LORA' in layer)
        frequency[method]['pct20_50_count_lora'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.2 <= int(layer.split(':')[0]) < total_layers *
            0.5 and ":" in layer and 'LORA' in layer)
        frequency[method]['pct50_80_count_lora'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.5 <= int(layer.split(':')[0]) < total_layers *
            0.8 and ":" in layer and 'LORA' in layer)
        frequency[method]['pct80_100_count_lora'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.8 <= int(layer.split(':')[0]) and ":" in layer and 'LORA' in layer)
        
        frequency[method]['pct20_count_ada'] = sum(
            freq for layer, freq in layer_stats.items()
            if int(layer.split(':')[0]) < total_layers * 0.2 and ":" in layer and 'ADAPTER' in layer)
        frequency[method]['pct20_50_count_ada'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.2 <= int(layer.split(':')[0]) < total_layers *
            0.5 and ":" in layer and 'ADAPTER' in layer)
        frequency[method]['pct50_80_count_ada'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.5 <= int(layer.split(':')[0]) < total_layers *
            0.8 and ":" in layer and 'ADAPTER' in layer)
        frequency[method]['pct80_100_count_ada'] = sum(
            freq for layer, freq in layer_stats.items()
            if total_layers * 0.8 <= int(layer.split(':')[0]) and ":" in layer and 'ADAPTER' in layer)
    return frequency

def transform_prune_dict(prune_dict):
    target_dict = {
        'lora_20': [],
        'lora_50': [],
        'lora_80': [],
        'lora_100': [],
        'adapter_20': [],
        'adapter_50': [],
        'adapter_80': [],
        'adapter_100': []
    }
    max_values = {
        'lora_20': float('-inf'),
        'lora_50': float('-inf'),
        'lora_80': float('-inf'),
        'lora_100': float('-inf'),
        'adapter_20': float('-inf'),
        'adapter_50': float('-inf'),
        'adapter_80': float('-inf'),
        'adapter_100': float('-inf')
    }
    
    for method, stats in prune_dict.items():
        # 更新每个层的最大值和对应的方法
        if stats['pct20_count_lora'] > max_values['lora_20']:
            max_values['lora_20'] = stats['pct20_count_lora']
            target_dict['lora_20'] = [method]
        elif stats['pct20_count_lora'] == max_values['lora_20']:
            target_dict['lora_20'].append(method)

        if stats['pct20_50_count_lora'] > max_values['lora_50']:
            max_values['lora_50'] = stats['pct20_50_count_lora']
            target_dict['lora_50'] = [method]
        elif stats['pct20_50_count_lora'] == max_values['lora_50']:
            target_dict['lora_50'].append(method)

        if stats['pct50_80_count_lora'] > max_values['lora_80']:
            max_values['lora_80'] = stats['pct50_80_count_lora']
            target_dict['lora_80'] = [method]
        elif stats['pct50_80_count_lora'] == max_values['lora_80']:
            target_dict['lora_80'].append(method)

        if stats['pct80_100_count_lora'] > max_values['lora_100']:
            max_values['lora_100'] = stats['pct80_100_count_lora']
            target_dict['lora_100'] = [method]
        elif stats['pct80_100_count_lora'] == max_values['lora_100']:
            target_dict['lora_100'].append(method)

        if stats['pct20_count_ada'] > max_values['adapter_20']:
            max_values['adapter_20'] = stats['pct20_count_ada']
            target_dict['adapter_20'] = [method]
        elif stats['pct20_count_ada'] == max_values['adapter_20']:
            target_dict['adapter_20'].append(method)

        if stats['pct20_50_count_ada'] > max_values['adapter_50']:
            max_values['adapter_50'] = stats['pct20_50_count_ada']
            target_dict['adapter_50'] = [method]
        elif stats['pct20_50_count_ada'] == max_values['adapter_50']:
            target_dict['adapter_50'].append(method)

        if stats['pct50_80_count_ada'] > max_values['adapter_80']:
            max_values['adapter_80'] = stats['pct50_80_count_ada']
            target_dict['adapter_80'] = [method]
        elif stats['pct50_80_count_ada'] == max_values['adapter_80']:
            target_dict['adapter_80'].append(method)

        if stats['pct80_100_count_ada'] > max_values['adapter_100']:
            max_values['adapter_100'] = stats['pct80_100_count_ada']
            target_dict['adapter_100'] = [method]
        elif stats['pct80_100_count_ada'] == max_values['adapter_100']:
            target_dict['adapter_100'].append(method)
            
    for method, stats in prune_dict.items():
        # find the max value in each pruning strategy
        max_item = max(stats['pct20_count_lora'], stats['pct20_50_count_lora'], stats['pct50_80_count_lora'], stats['pct80_100_count_lora'],
                    stats['pct20_count_ada'], stats['pct20_50_count_ada'], stats['pct50_80_count_ada'], stats['pct80_100_count_ada'])
        # save the pruning strategy to the corresponding list in the target dict
        if max_item == stats['pct20_count_lora'] and method not in target_dict['lora_20']:
            target_dict['lora_20'].append(method)
        if max_item == stats['pct20_50_count_lora'] and method not in target_dict['lora_50']:
            target_dict['lora_50'].append(method)
        if max_item == stats['pct50_80_count_lora'] and method not in target_dict['lora_80']:
            target_dict['lora_80'].append(method)
        if max_item == stats['pct80_100_count_lora'] and method not in target_dict['lora_100']:
            target_dict['lora_100'].append(method)

        if max_item == stats['pct20_count_ada'] and method not in target_dict['adapter_20']:
            target_dict['adapter_20'].append(method)
        if max_item == stats['pct20_50_count_ada'] and method not in target_dict['adapter_50']:
            target_dict['adapter_50'].append(method)
        if max_item == stats['pct50_80_count_ada'] and method not in target_dict['adapter_80']:
            target_dict['adapter_80'].append(method)
        if max_item == stats['pct80_100_count_ada'] and method not in target_dict['adapter_100']:
            target_dict['adapter_100'].append(method)
    return target_dict


def main(log_root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    stats = parse_log_files(log_root_folder)
    frequency = fre_cnt(stats)
    print(frequency)
    
    result = transform_prune_dict(frequency)
    print(f"result: {result}")


if __name__ == "__main__":
    log_folder = "results/layer_cnt_src"  # replace with the path to the folder containing the log files
    output_folder = "results/frequency_cnt"  # replace with the path to the output folder
    main(log_folder, output_folder)
