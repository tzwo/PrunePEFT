"""
TrainerWithGrad records gradients and activation when training.
which should be equivalent to Trainer in transformers.
"""

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, Adafactor
from torch.nn import CrossEntropyLoss
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, mean_squared_error

from pruning_methods import *
from utils.save_submit import saveInstance


class TrainerWithGrad:

    def __init__(self,
                 model,
                 args,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 compute_metrics,
                 callbacks,
                 tokenizer,
                 loss_fn=CrossEntropyLoss(),
                 device_ids=[0],
                 saver=...):
        self.device = torch.device("cuda:" +
                                   ','.join(map(str, device_ids)) if torch.cuda
                                   .is_available() else "cpu")
        # self.model = torch.nn.DataParallel(
        #     model.to(self.device), device_ids=device_ids)
        self.model = model.to(self.device)
        print('RUNNING ON DEVICE: ', self.device)
        # self.device='cuda'
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.tokenizer = tokenizer

        self.epoch_num = args.num_train_epochs
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size
        self.eval_strategy = args.evaluation_strategy
        self.saver = saver

        self.dummy_input = None

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn)
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn)

        self.loss_fn = loss_fn

        print('finished init')

    def get_loss(self, logits, labels):
        if labels.dtype == torch.float64:
            return self.loss_fn(logits.float(), labels.float())
        else:
            return self.loss_fn(logits, labels)

    def train(self, use_original=False):
        self.model = self.model.to(self.device)
        optimizer_grouped_parameters = [{
            "params": [
                param for name, param in self.model.named_parameters()
                if "lora" in name
            ],
            "lr": self.args.learning_rate['lora'],
        }, {
            "params": [
                param for name, param in self.model.named_parameters()
                if "adapter" in name
            ],
            "lr": self.args.learning_rate['adapter'],
        }, {
            "params": [
                param for name, param in self.model.named_parameters()
                if 'heads' in name
            ],
            "lr": 2e-4
        }]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters, weight_decay=self.args.weight_decay)
        optimizer = Adafactor(
            optimizer_grouped_parameters, weight_decay=self.args.weight_decay)


        total_steps = len(self.train_dataloader) * self.args.num_train_epochs


        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=total_steps / 100,
        #     num_training_steps=total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps / 100,
            num_training_steps=total_steps,
            num_cycles=0.5)

        activations = {}
        intermediate_results = []

        # hook function to save activations
        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output[0]
            return hook

        # apply hook to all layers
        hooks = []
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    hooks.append(
                        module.register_forward_hook(
                            save_activation(name + "." + param_name)))
                    break

        for epoch in range(self.epoch_num):
            self.model.train()
            gradients = {}
            for batch in tqdm(
                    self.train_dataloader,
                    desc=f"Training epoch {epoch+1}",
                    unit="batch"):
                batch = {
                    key: value.to(self.device) for key, value in batch.items()
                }
                outputs = self.model(**batch)
                loss = self.get_loss(outputs.logits, batch['label'])
                optimizer.zero_grad()
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = gradients.setdefault(
                            name, 0) + param.grad.detach().cpu().numpy()

                optimizer.step()
                scheduler.step()
            if self.eval_strategy == 'epoch':
                intermediate_results.append(self.evaluate())
            if self.saver != ...:
                submit_res = self.evaluate_on_testset()
                self.saver.save_submit(submit_res, epoch)

        for hook in hooks:
            hook.remove()

        return gradients, activations, intermediate_results

    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_count = 0, 0, len(self.val_dataset)
        y_true = []
        y_pred = []
        for batch in self.val_dataloader:
            with torch.no_grad():
                batch = {
                    key: value.to(self.device) for key, value in batch.items()
                }
                outputs = self.model(**batch)
                loss = self.get_loss(outputs.logits, batch['label'])
                total_loss += loss.item()

                y_true.extend(batch['label'].cpu().numpy())
                if torch.is_floating_point(batch['label'][0]):
                    y_pred.extend(outputs.logits.cpu().flatten().tolist())
                else:
                    y_pred.extend(outputs.logits.argmax(dim=-1).cpu().numpy())

                # print('type:', batch['label'][0].cpu())
                # print('output.logit:', outputs.logits)
                # print('y_true:', y_true)
                # print('y_pred:', y_pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = 0.0
        f1 = 0.0
        mcc = 0.0
        mse = 0.0

        try:
            accuracy = accuracy_score(y_true, y_pred)
        except Exception as e:
            pass

        try:
            f1 = f1_score(y_true, y_pred, average='macro')
        except:
            pass

        try:
            # conf_matrix = confusion_matrix(y_true, y_pred)
            pass
        except:
            pass

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            pass
        try:
            mse = mean_squared_error(y_true, y_pred)
        except:
            pass

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Macro F1 Score: {f1:.4f}')
        # print('Confusion Matrix:', conf_matrix)
        print('mcc:', mcc)

        print(
            f'Epoch {self.epoch_num+1} || Validation loss: {total_loss/total_count:.6f} || Validation accuracy: {accuracy:.6f}'
        )
        return {
            'eval_loss': total_loss / total_count,
            'eval_accuracy': accuracy,
            'accuracy': accuracy,
            'f1': f1,
            # 'conf_matfix': conf_matrix,
            'mcc': mcc,
            'mse': mse,
        }

    def evaluate_on_testset(self):
        self.model.eval()
        idxs = []
        y_pred = []
        for batch in self.test_dataloader:
            with torch.no_grad():
                batch = {
                    key: value.to(self.device) for key, value in batch.items()
                }
                outputs = self.model(**batch)

                idxs.extend(batch['idx'].cpu().numpy())
                if torch.is_floating_point(batch['label'][0]):
                    y_pred.extend(outputs.logits.cpu().flatten().tolist())
                else:
                    y_pred.extend(outputs.logits.argmax(dim=-1).cpu().numpy())

        idxs = np.array(idxs)
        y_pred = np.array(y_pred)

        return pd.DataFrame({'IDs': idxs, 'labels': y_pred})

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.tensor([i['input_ids'] for i in batch])
        attention_mask = torch.tensor([i['attention_mask'] for i in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'idx': torch.tensor([item['idx'] for item in batch]),
            'label': torch.tensor([item['label'] for item in batch])
        }
