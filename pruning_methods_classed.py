'''

This file contains functions for model pruning, including getting trainable parameters, 
grouping parameters by prefix, finding groups with the smallest values, and more.

'''
import re
import numpy as np
import math


def get_trainable_parameters(model):
    """Get the list of trainable parameter names in the model"""
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    return names


class PruneModel:
    scores = []
    components = []
    weights = {}
    methods = [
        'zeros', 'values_below_threshold', 'minimum_weight', 'gradient',
        'activation', 'snip'
    ]
    methods = ['values_below_threshold', 'gradient', 'activation', 'snip']
    last_accuracy = -1
    last_predict = -1

    def __init__(self, method_with_blocks) -> None:
        for method in self.methods:
            self.weights[method] = 1.0
        # self.weights['snip']=1.5
        self.method_with_blocks = method_with_blocks

    def prune_model(self,
                    model,
                    task_name='',
                    opts=['lora'],
                    p_method='zeros',
                    top_p=3,
                    print_names=False,
                    gradients=None,
                    hessians=None,
                    activations=None,
                    accuracy=0.5):
        """Prune the model's parameters"""
        modelname = model.config.model_type
        names = get_trainable_parameters(model)
        groups = PruneModel.group_parameters_by_prefix(
            names,
            opts=opts,
            print_names=print_names,
            task_name=task_name,
            model_name=modelname)

        if p_method == 'optimizer':
            # opt
            if self.last_accuracy == -1:
                self.last_accuracy = accuracy
                delta = -1
            else:
                delta = (accuracy - self.last_accuracy)
                self.last_accuracy = accuracy

            if delta != -1:
                for key in self.components[self.last_predict].keys():
                    self.weights[key] += delta * self.components[
                        self.last_predict][key]

                def softmax_dict(values_dict):
                    values = list(values_dict.values())
                    exp_values = [math.exp(v) for v in values]
                    sum_exp_values = sum(exp_values)
                    softmax_dict = {
                        key: exp_val / sum_exp_values
                        for key, exp_val in zip(values_dict.keys(), exp_values)
                    }
                    return softmax_dict

                probs = softmax_dict(self.weights)
            else:
                probs = self.weights
            print('delta:', delta)
            print('weights', self.weights)
            # calculate
            self.scores = [0] * 24
            self.components = [{} for _ in range(24)]
            top_p = 3
            for method in self.methods:
                sorted_groups = self.find_group_with_most_small_values(
                    groups, model, method, top_p, gradients, activations)
                for i in range(top_p):
                    match = re.search(r'layer(s)?\.(\d+)', sorted_groups[i][0])
                    if match:
                        layer_number = match.group(2)
                        self.components[int(
                            layer_number)][method] = probs[method]
            print(self.components)
            for layer in range(24):
                for key in self.methods:
                    if key in self.components[layer]:
                        self.scores[layer] += self.components[layer][key]
            max_index = self.scores.index(max(self.scores))
            self.last_predict = max_index
            return max_index, 'LORA'
        else:
            sorted_groups = self.find_group_with_most_small_values(
                groups, model, p_method, top_p, gradients, activations)
        # remove_layers(sorted_groups, model)
        if print_names:
            for group_info in sorted_groups:
                group, _, small_values = group_info
                print(
                    f"Pruned group: {group}, with {small_values} small values.")

        ret = []
        try:
            if p_method == 'block' or p_method == 'block_mixed':
                turn = 8
            elif p_method == 'block_dynamic':
                turn = 12
            else:
                turn = 3
            for i in range(turn):
                match = re.search(r'layer(s)?\.(\d+)', sorted_groups[i][0])
                if match:
                    layer_number = match.group(2)
                else:
                    layer_number = -1
                if 'lora' in sorted_groups[i][0]:
                    layer_type = 'LORA'
                else:
                    layer_type = 'ADAPTER'
                ret.append({
                    'layer_number': layer_number,
                    'layer_type': layer_type
                })
        except Exception as e:
            print(e)
        return ret

    @staticmethod
    def group_parameters_by_prefix(names,
                                   opts=[],
                                   print_names=False,
                                   task_name='',
                                   model_name=''):
        """Group parameters based on prefixes"""
        if model_name == 'roberta':
            v_name = 'query.'
            q_name = 'value.'
        else:
            v_name = 'q_proj.'
            q_name = 'v_proj.'
        groups = {}
        names = [
            name for name in names
            if task_name in name and 'head' not in name and any(
                opt in name for opt in opts)
        ]
        for name in names:
            prefix = name.split(task_name)[0]
            prefix = prefix.replace(v_name, '').replace(q_name, '')
            if prefix in groups:
                groups[prefix].append(name)
            else:
                groups[prefix] = [name]
        if print_names:
            for prefix, names in groups.items():
                print(f"{prefix}:")
                for name in names:
                    print(f"  {name}")
        return groups

    def find_group_with_most_small_values(self,
                                          groups,
                                          model,
                                          p_method,
                                          top_p=1,
                                          gradients=None,
                                          activations=None,
                                          methods_list=[
                                              'zeros',
                                              'gradient', 'values_below_threshold',
                                              'minimum_weight', 
                                              'activation', 'snip'
                                          ]):
        """Find groups with the smallest values or smallest summed gradients"""
        group_values = []
        # prefix = 'module.'
        prefix = ''
        if p_method == 'zeros':
            for group, names in groups.items():
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                num_zeros = sum((model.state_dict()[name] == 0).sum().item()
                                for name in names)
                ratio_zeros = num_zeros / total_params if total_params > 0 else 0
                group_values.append((group, names, ratio_zeros))

        elif p_method == 'values_below_threshold':
            threshold = 0.000001  # Adjust the threshold as needed
            for group, names in groups.items():
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                num_values_below_threshold = sum(
                    (model.state_dict()[name].abs() < threshold).sum().item()
                    for name in names)
                ratio_below_threshold = num_values_below_threshold / total_params if total_params > 0 else 0
                group_values.append((group, names, ratio_below_threshold))

        elif p_method == 'minimum_weight':
            for group, names in groups.items():
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                min_weight = min((model.state_dict()[name].pow(2).sum() /
                                  model.state_dict()[name].numel()).item()
                                 for name in names)
                avg_min_weight = min_weight / total_params if total_params > 0 else 0
                group_values.append((group, names, avg_min_weight))

        elif p_method == 'gradient':
            if gradients is None:
                raise ValueError(
                    "Gradients must be provided for 'gradient' p_method")
            for group, names in groups.items():
                # print(groups)
                # print(gradients)
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                total_gradient = sum(
                    np.abs(gradients[prefix + name]).sum().item()
                    for name in names)
                avg_gradient = total_gradient / total_params if total_params > 0 else 0
                group_values.append((group, names, avg_gradient))

        elif p_method == 'activation':
            activation_score = 0
            if activations is None:
                raise ValueError(
                    "Activations must be provided for 'activation' p_method")
            for group, names in groups.items():
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                for name in names:
                    activation = activations[prefix + name]
                    activation_score += activation.abs().sum().item()
                    break
                avg_activation_score = activation_score / total_params if total_params > 0 else 0
                group_values.append((group, names, avg_activation_score))

        elif p_method == 'snip':
            if gradients is None:
                raise ValueError(
                    "Gradients must be provided for 'snip' p_method")

            # Calculate the gradient magnitudes
            gradient_magnitudes = {}
            for name in gradients:
                gradient_magnitudes[name] = np.abs(gradients[name])

            # Normalize the gradients
            norm_gradients = {}
            for name, grad in gradient_magnitudes.items():
                weight = model.state_dict()[
                    name].cpu().numpy()  # Convert weight to numpy array
                norm_gradients[name] = grad * weight

            # Sum normalized gradients for each group
            for group, names in groups.items():
                total_params = sum(
                    model.state_dict()[name].numel() for name in names)
                total_snip = sum(norm_gradients[prefix + name].sum().item()
                                 for name in names)
                avg_snip = total_snip / total_params if total_params > 0 else 0
                group_values.append((group, names, avg_snip))

        elif p_method == 'mixed':
            all_group_values = []
            methods = methods_list

            for method in methods:
                # Call the function recursively for each method
                method_group_values = self.find_group_with_most_small_values(
                    groups, model, method, top_p, gradients, activations)
                max_value = (max(value for _, _, value in method_group_values) +
                             1e-5) if method_group_values else 1
                method_group_values = [
                    (group, names, value / max_value)
                    for group, names, value in method_group_values
                ]
                all_group_values.append(method_group_values)

            combined_scores = {}
            for method_values in all_group_values:
                for rank, (group, names, value) in enumerate(method_values):
                    # the weight declines as the rank drops
                    # weight = 10 - 2 * rank
                    # if weight <= 0:
                    #     break

                    weight = 5 / (1 + np.exp(rank - 3))

                    softmax_value = np.exp(value) / np.sum(
                        np.exp([v[2] for v in method_values]))  
                    if group in combined_scores:
                        combined_scores[group][0] += softmax_value * weight
                    else:
                        combined_scores[group] = [
                            softmax_value * weight, names
                        ]  

            group_values = [(group, names, score)
                            for group, (score,
                                        names) in combined_scores.items()]

        elif p_method == 'block':
            blocks = PruneModel.split_group(groups)

            group_values = []

            for block, method in self.method_with_blocks.items():
                method_group_values = self.find_group_with_most_small_values(
                        blocks[block], model, method[0], top_p, gradients, activations)
                group_values.append(method_group_values[0])

        elif p_method == 'block_mixed':

            blocks = PruneModel.split_group(groups)

            group_values = []

            for block, methods in self.method_with_blocks.items():
                method_group_values = self.find_group_with_most_small_values(
                    blocks[block],
                    model,
                    "mixed",
                    top_p,
                    gradients,
                    activations,
                    methods_list=methods)
                group_values.append(method_group_values[0])
                
        elif p_method == 'block_dynamic':

            blocks = PruneModel.split_group(groups)

            group_values = []

            for block, methods in self.method_with_blocks.items():
                method_group_values = self.find_group_with_most_small_values(
                    blocks[block],
                    model,
                    "mixed",
                    top_p,
                    gradients,
                    activations,
                    methods_list=methods)
                
                exp_values = np.exp([v[2] for v in method_group_values])
                sum_exp_values = np.sum(exp_values)

                for rank, (group, names, value) in enumerate(method_group_values):
                    softmax_value = np.exp(value) / sum_exp_values
                    method_group_values[rank] = (group, names, softmax_value)
                group_values +=  method_group_values
        sorted_groups = sorted(
            group_values,
            key=lambda x: x[2],
            reverse=(p_method in ['zeros', 'values_below_threshold',
                                  'mixed', 'block_dynamic']))[:top_p]
        return sorted_groups

    @staticmethod
    def remove(path, model, methods):
        """Remove modules or attributes at the specified path"""
        path = path.split('.')
        module = model
        for part in path:
            if part.isdigit():
                part = int(part)
                module = module[part]
            elif part == methods:
                module = delattr(module, part)
                break
            else:
                module = getattr(module, part)

    @staticmethod
    def remove_layers(groups, model):
        """Remove layers in the specified groups"""
        for group_info in groups:
            group_name, names, _ = group_info
            if 'lora' in group_name:
                for name in names:
                    if 'lora_B' in name:
                        continue
                    PruneModel.remove(name, model, 'loras')
            elif 'adapter' in group_name:
                PruneModel.remove(group_name[:-1], model, 'adapters')

    @staticmethod
    def split_group(group):
        """
        Divide the given group by LoRA and adapter types, as well as the number of layers.

        Args:
            group: The dictionary to be divided.

        Returns:
            A list containing multiple dictionaries after division.
        """

        lora_group = {k: v for k, v in group.items() if 'lora' in k}
        adapter_group = {k: v for k, v in group.items() if 'adapter' in k}

        max_layer = max(int(k.split('.')[3]) for k in group.keys())

        def split_by_layer(sub_group, start, end):
            return {
                k: v
                for k, v in sub_group.items()
                if start <= int(k.split('.')[3]) <= end
            }

        result = {
            'lora_20': {},
            'lora_50': {},
            'lora_80': {},
            'lora_100': {},
            'adapter_20': {},
            'adapter_50': {},
            'adapter_80': {},
            'adapter_100': {}
        }
        for item in lora_group:
            if int(item.split('.')[3]) < max_layer * 0.2:
                result['lora_20'][item] = lora_group[item]
            elif int(item.split('.')[3]) < max_layer * 0.5:
                result['lora_50'][item] = lora_group[item]
            elif int(item.split('.')[3]) < max_layer * 0.8:
                result['lora_80'][item] = lora_group[item]
            else:
                result['lora_100'][item] = lora_group[item]

        for item in adapter_group:
            if int(item.split('.')[3]) < max_layer * 0.2:
                result['adapter_20'][item] = adapter_group[item]
            elif int(item.split('.')[3]) < max_layer * 0.5:
                result['adapter_50'][item] = adapter_group[item]
            elif int(item.split('.')[3]) < max_layer * 0.8:
                result['adapter_80'][item] = adapter_group[item]
            else:
                result['adapter_100'][item] = adapter_group[item]

        return result
