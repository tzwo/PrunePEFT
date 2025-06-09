"""The main function that runs the model
It can be used by 2 ways:
1. run 'python run_model.py --lora 64' in the terminal
2. use PEFTModel.run(args/configs) in other python files
"""

import re

import numpy as np
import torch
import torch.nn as nn
from adapters import (
    ConfigUnion,
    LoRAConfig,
    SeqBnConfig,
)
from transformers import RobertaTokenizer, TrainingArguments, AutoTokenizer, TrainerCallback, Trainer, AdamW, get_cosine_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, Adafactor
from pruning_methods import get_trainable_parameters, group_parameters_by_prefix
from sklearn.metrics import f1_score
from src.trainer_with_grad import TrainerWithGrad
from src.dora import lora2dora
import random

from utils.regression_head import CustomRegressionHead
from utils.save_submit import saveInstance


def reset_seed():
    # set fixed seed
    random.seed(42)
    np.random.seed(42)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class GradientCaptureCallback(TrainerCallback):

    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} gradients:")
        for name, grad in self.model_trainer.gradients.items():
            print(f"{name}: {grad}")


class PEFTModel:
    """Model wrapper for PEFT

    Args:
        configs (dict): a dict of configs.
        trainer (Trainer): a trainer for model fine-tuning.
        dataset (Dataset): the training dataset.
        model_name (str): model name.
        task_name (str): task name.
        model (nn.Module): model to fune-tuning.
        tokenizer (RobertaTokenizer): tokenizer of language model.
    """

    def __init__(self, configs, dataset=None):
        """
        Args:
            configs: a dict of configs
            dataset: a dataset object
        """
        self.trainer = None
        self.configs = configs

        self.model_name=self.configs['BACKBONE']
        self.task_name = "mytask"

        self.dataset = dataset
        # reset_seed()
        if self.model_name == 'roberta-large':
            tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        elif self.model_name == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                token=...) # you should add your own token here
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('Model not supported')

        self.instructs = 0
        self.epochs = 1
        self.lr = 2e-5
        self.gradients = {}

        # calculate the number of labels
        if hasattr(dataset["train"].features["label"], "num_classes"):
            num_labels = dataset["train"].features["label"].num_classes
        else:
            num_labels = len(set(dataset["train"]["label"]))
        print("number of label classes:", num_labels)

        if not 'LOSS' in configs or configs['LOSS'] == 'cross_entropy':
            self.model.add_classification_head(
                self.task_name, num_labels=num_labels)
        elif configs['LOSS'] == 'mse':
            self.model.register_custom_head("regression_head",
                                            CustomRegressionHead)
            self.model.add_custom_head(
                head_type="regression_head", head_name=self.task_name)
        else:
            raise ('not implemented loss')


        lora_config = None
        adapter_config = None
        dora_config = None
        peft_config = None
        if configs.get("base_lora"):
            lora_config = LoRAConfig(
                r=next(x for x in configs["base_lora"]["ranks"] if x != 0),
                alpha=next(x for x in configs["base_lora"]["ranks"] if x != 0),
                dropout=0.1,
            )
        if configs.get("base_adapter"):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs["base_adapter"]["bn"] if x != 0),)
        if configs.get("LORA"):
            lora_config = LoRAConfig(
                r=next(x for x in configs["LORA"] if x != 0),
                alpha=next(x for x in configs["LORA"] if x != 0),
                dropout=0.1,
                leave_out=[i for i, x in enumerate(configs["LORA"]) if x == 0],
            )
        if configs.get("DORA"):
            dora_config = LoRAConfig(
                r=next(x for x in configs["DORA"] if x != 0),
                alpha=next(x for x in configs["DORA"] if x != 0),
                dropout=0.1,
                leave_out=[i for i, x in enumerate(configs["DORA"]) if x == 0],
            )
        if configs.get("ADAPTER"):
            adapter_config = SeqBnConfig(
                reduction_factor=next(x for x in configs["ADAPTER"] if x != 0),
                leave_out=[
                    i for i, x in enumerate(configs["ADAPTER"]) if x == 0
                ],
                dropout=0.1,
            )

        if lora_config and adapter_config:
            peft_config = ConfigUnion(lora_config, adapter_config)
        elif lora_config:
            peft_config = lora_config
        elif dora_config:
            peft_config = dora_config
        elif adapter_config:
            peft_config = adapter_config
        else:
            assert 0

        self.model.add_adapter("my_module", config=peft_config)
        self.model.train_adapter("my_module")
        if configs.get("DORA"):
            lora2dora(self.model)
        self.model = self.model.half()

        if configs.get("LORA"):
            names = get_trainable_parameters(self.model)
            groups = group_parameters_by_prefix(
                names, opts="lora", task_name="my_module")
            sorted_groups = sorted(groups.items())
            sorted_groups = [name[1] for name in sorted_groups]

            ranks = [r for r in configs["LORA"] if r != 0]
            for group, r in zip(sorted_groups, ranks):
                self.set_peft_group(group, "set", r)

        if configs.get("ADAPTER"):
            names = get_trainable_parameters(self.model)
            groups = group_parameters_by_prefix(
                names, opts="adapter", task_name="my_module")
            sorted_groups = sorted(groups.items())
            sorted_groups = [name[1] for name in sorted_groups]

            ranks = [r for r in configs["ADAPTER"] if r != 0]
            for group, r in zip(sorted_groups, ranks):
                self.set_peft_group(group, "set", r)

        if configs.get('EPOCHS'):
            self.epochs = configs['EPOCHS']
        else:
            self.epochs = 1

        if configs.get("INSTRUCTS"):
            self.instructs = 1

        if configs.get("LOSS"):
            self.loss_type = configs["LOSS"]
            if (self.loss_type == "cross_entropy"):
                self.loss_fn = nn.CrossEntropyLoss()
            elif (self.loss_type == "mse"):
                self.loss_fn = nn.MSELoss()
                print('mse')
            else:
                raise (f"loss type {self.loss_type} not supported")
        else:
            self.loss_type = "cross_entropy"
            self.loss_fn = nn.CrossEntropyLoss()

        if configs.get("LORA_LR"):
            self.lora_lr = float(configs["LORA_LR"])
        else:
            self.lora_lr = 1e-5

        if configs.get("ADAPTER_LR"):
            self.adapter_lr = float(configs["ADAPTER_LR"])
        else:
            self.adapter_lr = 1e-6

    def run(self,
            device_ids=[0],
            submit_flag=False,
            val_flag=True,
            saver=...,
            use_original_trainer=False):
        """tokenize the dataset and train the model"""
        assert self.dataset is not None
        assert self.model is not None
        assert self.tokenizer is not None

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = np.sum(preds == labels) / len(labels)
            f1 = f1_score(labels, preds, average='weighted')
            final_score = 0.5 * acc + 0.5 * f1
            result = {"accuracy": acc, "f1": f1, "final_score": final_score}
            print(result)
            return result

        loss_fn = self.loss_fn

        def compute_loss(outputs, labels, num_items_in_batch):
            logits = outputs['logits']
            global loss_fn
            return loss_fn(logits, labels)

        import os

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="./logs",
            learning_rate={
                'lora': self.lora_lr,
                'dora': self.lora_lr,
                'adapter': self.adapter_lr,
            },
            evaluation_strategy="epoch",
            # compute_loss_func=compute_loss,
        )
        self.trainer = TrainerWithGrad(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            val_dataset=self.dataset['validation'],
            test_dataset=self.dataset['test'],
            compute_metrics=compute_metrics,
            callbacks=[GradientCaptureCallback(self)],
            tokenizer=self.tokenizer,
            loss_fn=self.loss_fn,
            device_ids=device_ids,
            saver=saver,
        )
        optimizer_grouped_parameters = [{
            "params": [
                param for name, param in self.model.named_parameters()
                if "lora" in name
            ],
            "lr": training_args.learning_rate['lora'],
        }, {
            "params": [
                param for name, param in self.model.named_parameters()
                if "adapter" in name
            ],
            "lr": training_args.learning_rate['adapter'],
        }, {
            "params": [
                param for name, param in self.model.named_parameters()
                if 'heads' in name
            ],
            "lr": 2e-4
        }]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     weight_decay=training_args.weight_decay)
        batch_size = 16
        total_steps = len(self.dataset['train']) * self.epochs // batch_size
        optimizer = Adafactor(optimizer_grouped_parameters, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps / 10,
            num_training_steps=total_steps,
            num_cycles=0.5)

        # ori_training_args = TrainingArguments(
        #     output_dir="./results",
        #     num_train_epochs=self.epochs,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=8,
        #     warmup_ratio=0.1,
        #     weight_decay=0.01,
        #     logging_dir="./logs",
        #     # learning_rate=self.lora_lr,
        #     evaluation_strategy="epoch",
        #     save_steps=1000000,
        #     # logging_steps=100,
        #     label_names=['label'],
        #     # remove_unused_columns= False, # 在compute_loss 时需要额外输入
        #     include_inputs_for_metrics= True, # compute_metrics 时需要原始输出来计算评价指标
        # )
        # ori_trainer=Trainer(
        #     model=self.model,
        #     args=ori_training_args,
        #     train_dataset=self.dataset['train'],
        #     eval_dataset=self.dataset['validation'],
        #     compute_metrics=compute_metrics,
        #     tokenizer=self.tokenizer,
        #     optimizers=(optimizer, scheduler),
        # )

        # Register hooks to capture gradients
        if not use_original_trainer:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(
                        lambda grad, name=name: self.gradients.update(
                            {name: grad.clone().cpu().detach().numpy()}))

        if use_original_trainer:
            print('USING ORIGINAL TRAINER')
            ori_trainer.train()
            gradients, activations, intermediate_results = ..., ..., ...
        else:
            print('USING CUSTOM TRAINER')
            gradients, activations, intermediate_results = self.trainer.train(
                use_original=use_original_trainer)
        if val_flag:
            metrics = self.trainer.evaluate()
            print(metrics)
        else:
            metrics = {}
        if submit_flag:
            submit_res = self.trainer.evaluate_on_testset()
        else:
            submit_res = ...

        print(metrics)
        metrics['intermediate_results'] = intermediate_results

        return metrics, gradients, activations, submit_res

    def add_dataset(self, dataset):
        """add a dataset"""
        self.dataset = dataset

    def set_peft_group(self, group, index, value=0):
        """set the size of a PEFT module
        index='double' means to double the rank/bn of LoRA or adapter
        index='half' means to half the rank/bn of LoRA or adapter
        index='remove' means to remove
        index='set' means to set the rank to value
        """
        # exec(
        # "module=self.model." + group[0], locals()
        # )  # save the var into locals(), which cannot be repetitive in function
        # mo = locals()["module"]

        # NOTE: change to this
        self.model: nn.Module
        mo = self.model.get_parameter(
            group[0])  # or mo = self.model.__getattr__(group[0])
        group = [re.sub(r"\.(\d+)", r"[\1]", name) for name in group]

        origin_emb_size = mo.size()[1]
        origin_rank_size = mo.size()[0]

        target_rank_size = origin_rank_size
        if index == "double":
            target_rank_size *= 2
        elif index == "half":
            target_rank_size //= 2
        elif index == "remove":
            target_rank_size = 0
        elif index == "set":
            target_rank_size = value
        else:
            assert 0

        if origin_rank_size == target_rank_size:
            return
        if target_rank_size == 0:
            assert 0

        # Maybe we do not need initial when reinitialed the whole model
        for name in [name for name in group if "lora_A" in name]:
            # weights = torch.zeros(target_rank_size, origin_emb_size)
            # nn.init.kaiming_uniform_(
            #     weights, a=math.sqrt(5)
            # )  # TODO: use rand will unable to train, use kaiming init is weaker than default(0.74 vs 0.77 on rotten_tomatoes 4 epoch)

            weights = torch.empty(target_rank_size, origin_emb_size)
            torch.nn.init.normal(
                weights, mean=0, std=1 / pow(origin_emb_size, 0.5))

            # torch.nn.init.xavier_normal_(weights)
            # torch.nn.init.kaiming_normal_(weights, nonlinearity='relu')
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

        for name in [name for name in group if "lora_B" in name]:
            weights = torch.zeros(origin_emb_size, target_rank_size)
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")
        for name in [
                name for name in group
                if "adapter_down" in name and "weight" in name
        ]:
            weights = torch.empty(target_rank_size, origin_emb_size)
            torch.nn.init.normal(
                weights, mean=0, std=1 / pow(origin_emb_size, 0.5))

            # torch.nn.init.xavier_normal_(weights)
            # torch.nn.init.kaiming_normal_(weights, nonlinearity='relu')

            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

        for name in [
                name for name in group
                if "adapter_down" in name and "bias" in name
        ]:
            weights = torch.zeros(target_rank_size)
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

        for name in [
                name for name in group
                if "adapter_up" in name and "weight" in name
        ]:
            weights = torch.empty(origin_emb_size, target_rank_size)
            torch.nn.init.normal(
                weights, mean=0, std=1 / pow(origin_emb_size, 0.5))

            # torch.nn.init.xavier_normal_(weights)
            # torch.nn.init.kaiming_normal_(weights, nonlinearity='relu')
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

        for name in [
                name for name in group
                if "adapter_up" in name and "bias" in name
        ]:
            weights = torch.zeros(origin_emb_size)
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

    def half(self):
        self.model = self.model.half()
        return self

