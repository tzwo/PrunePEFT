'''
Get dataset and preprocess
split the dataset into train and test
'''

from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer


class PEFTDataset():
    '''
    '''
    dataset = None
    train_dataset = None
    test_dataset = None
    validation_dataset = None

    def __init__(self,
                 dataset_name,
                 task_name=None,
                 instructs=False,
                 test_size=1.0,
                 train_size=1.0,
                 shuffle=False,
                 model_name='roberta-large'):
        '''
        dataset_name: a string,
        task_name: a string, can be empty, only use in "glue/cola" etc.

        update: use local dataset
        '''
        dataset = load_from_disk(f'data/{dataset_name}_{task_name}')

        if shuffle:
            dataset = dataset.shuffle(seed=42)

        instruct_string = ""
        if (dataset_name == "glue"):
            if (task_name == "cola"):
                pass

        # split the dataset, dataset should have "train", "test", "validation"
        # if train_size/test_size<1.0, then the dataset will be split by rate
        # if train_size/test_size>1.0, then the dataset will be split by number

        original_train_size = len(dataset['train'])
        original_val_size = len(dataset['validation'])
        original_test_size = len(dataset['test'])

        if train_size <= 1.0:
            new_train_size = int(original_train_size * train_size)
        elif train_size > 1.0:
            new_train_size = min(int(train_size), original_train_size)

        if test_size <= 1.0:
            new_val_size = int(original_val_size * test_size)
            new_test_size = int(original_test_size * test_size)
        elif test_size > 1.0:
            new_val_size = min(int(test_size), original_val_size)
            new_test_size = min(int(test_size), original_test_size)

        if new_train_size < original_train_size:
            train_datast = dataset['train'].train_test_split(
                test_size=new_train_size)['test']
        else:
            train_datast = dataset['train']
        if new_val_size < original_val_size:
            val_dataset = dataset['validation'].train_test_split(
                test_size=new_val_size)['test']
        else:
            val_dataset = dataset['validation']
        if new_test_size < original_test_size:
            test_dataset = dataset['test'].train_test_split(
                test_size=new_test_size)['test']
        else:
            test_dataset = dataset['test']

        print(f'Loading dataset {dataset_name} {task_name}.')

        dataset = DatasetDict({
            'train': train_datast,
            'validation': val_dataset,
            'test': test_dataset
        })

        if instructs and not instruct_string:

            def add_prefix(example):
                example["text"] = instruct_string + example["text"]
                return example

            dataset = dataset.map(add_prefix)

        self.dataset = dataset
        self.max_length = 256
        # prepare tokenized dataset
        if model_name == 'roberta-large':
            tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        elif model_name == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                token=...) # you should add your own token here
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('Model not supported')

        columns_to_remove = ['idx', 'label']
        train_field = Dataset.from_dict({
            k: self.dataset['train'][k]
            for k in self.dataset['train'].column_names
            if k not in columns_to_remove
        })
        val_field = Dataset.from_dict({
            k: self.dataset['validation'][k]
            for k in self.dataset['validation'].column_names
            if k not in columns_to_remove
        })
        test_field = Dataset.from_dict({
            k: self.dataset['test'][k]
            for k in self.dataset['test'].column_names
            if k not in columns_to_remove
        })

        def tokenize_function(examples):
            return tokenizer(
                *examples.values(),
                padding='max_length',
                truncation=True,
                max_length=self.max_length)

        train_field = train_field.map(tokenize_function)
        val_field = val_field.map(tokenize_function)
        test_field = test_field.map(tokenize_function)
        print(test_field)
        self.train_field = train_field.add_column(
            "label", self.dataset['train']['label']).add_column(
                "labels", self.dataset['train']['label']).add_column(
                    "idx", self.dataset['train']['idx'])
        self.val_field = val_field.add_column(
            "label", self.dataset['validation']['label']).add_column(
                "labels", self.dataset['validation']['label']).add_column(
                    "idx", self.dataset['validation']['idx'])
        self.test_field = test_field.add_column(
            "label", self.dataset['test']['label']).add_column(
                "labels", self.dataset['test']['label']).add_column(
                    "idx", self.dataset['test']['idx'])

    def balanced_split(self, size=0.5):
        # 将train数据集变成平衡的size比例
        train_datast = self.dataset['train']
        val_dataset = self.dataset['validation']
        test_dataset = self.dataset['test']
        if size <= 1.0:
            positive_count = len([e for e in train_datast if e['label'] == 1
                                 ]) * size
            negative_count = len([e for e in train_datast if e['label'] == 0
                                 ]) * size
        else:
            positive_count = len([e for e in train_datast if e['label'] == 1
                                 ]) * size // train_datast.num_rows
            negative_count = len([e for e in train_datast if e['label'] == 0
                                 ]) * size // train_datast.num_rows

        # Balance the dataset by resampling
        positive_examples = [
            i for i, e in enumerate(train_datast) if e['label'] == 1
        ][:int(positive_count)]
        negative_examples = [
            i for i, e in enumerate(train_datast) if e['label'] == 0
        ][:int(negative_count)]
        print('origin size:', train_datast.num_rows)
        print('origin pos/neg rate: ', positive_count / negative_count)

        balanced_examples = train_datast.select(positive_examples +
                                                negative_examples)

        self.dataset = DatasetDict({
            'train': balanced_examples,
            'validation': val_dataset,
            'test': test_dataset
        }).shuffle(seed=42)

        ###
        positive_count = len([e for e in balanced_examples if e['label'] == 1
                             ]) * size
        negative_count = len([e for e in balanced_examples if e['label'] == 0
                             ]) * size
        print('balanced size:', balanced_examples.num_rows)
        print('balanced pos/neg rate: ', positive_count / negative_count)

        return self

    def get_dataset(self):
        # return the dataset
        print("Dataset preview:", self.dataset)
        return self.dataset

    def get_tokenized_dataset(self):
        # return the tokenized dataset
        # with origin columns, like labels and idx

        return DatasetDict({
            'train': self.train_field,
            'validation': self.val_field,
            'test': self.test_field
        })


