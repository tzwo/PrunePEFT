import os
import time

task_names = [
    'cola', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte',
    'sst2', 'stsb', 'wnli', 'ax'
]
super_task_names = [
    'boolq', 'cb', 'copa', 'rte', 'wic', 'multirc', 'record', 'wsc', 'axg'
]
names = [
    'CoLA.tsv', 'MNLI-m.tsv', 'MNLI-mm.tsv', 'MRPC.tsv', 'QNLI.tsv', 'QQP.tsv',
    'RTE.tsv', 'SST-2.tsv', 'STS-B.tsv', 'WNLI.tsv', 'AX.tsv'
]
super_names = [
    'BoolQ.jsonl', 'CB.jsonl', 'COPA.jsonl', 'MultiRC.jsonl', 'ReCoRD.jsonl',
    'RTE.jsonl', 'WiC.jsonl', 'WSC.jsonl', 'AX-b.jsonl', 'AX-g.jsonl'
]


class saveInstance():

    def __init__(self, dataset_name, task_name, prune_method, day_str,
                 time_str):
        # make submit dir

        # time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        # day_str = time.strftime('%Y-%m-%d', time.localtime())

        if 'SUPER_GLUE' == dataset_name:
            submit_dir = f'submit/{day_str}/submit_{time_str}/super_glue_{task_name}/{prune_method}/'
        else:
            submit_dir = f'submit/{day_str}/submit_{time_str}/glue_{task_name}/{prune_method}/'

        self.submit_dir = submit_dir
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.prune_method = prune_method

    def set_taskname(self, taskname):
        self.task_name = taskname

    def save_submit(self, submit_res, middle_res=-1):
        # save submit file to submit/{taskname}
        # GLUE and Super-GLUE
        submit_dir = self.submit_dir + f'epoch{middle_res}/'

        if not os.path.exists(submit_dir):
            os.makedirs(submit_dir)
        if 'SUPER_GLUE' == self.dataset_name:
            submit_name = super_names[super_task_names.index(self.task_name)]
        else:
            submit_name = names[task_names.index(self.task_name)]

        submit_res.to_csv(f'{submit_dir}/{submit_name}', sep='\t', index=False)
