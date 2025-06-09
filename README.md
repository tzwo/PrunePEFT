<p align="center">
  <h3 align="center">PrunePEFT code</h3>
  <p align="center">
    PrunePEFT: Adaptive Hybrid Pruning for Parameter-Efficient Fine-tuning of
Large Pre-trained Language Models
  </p>
</p>
![method](.\pic\method.png)

### Context

- [Context](#context)
- [Quick Start](#quick-start)
  - [create environment](#create-environment)
  - [dataset download](#dataset-download)
  - [file tree](#file-tree)
- [Run](#run)

### Quick Start

##### create environment

`conda create -n prunepeft python=3.10`

`conda activate prunepeft`

`pip install -r requirement.txt`

##### dataset download

run code in `glue_dataset.ipynb` to download GLUE dataset.

##### file tree

```
.
├── data
├── glue_dataset.ipynb
├── method_configs              # PEFT module configs
│   ├── adapter128.yaml
│   ├── adapter_lora.yaml
│   ├── dora.yaml
│   └── lora.yaml
├── model
├── outputs                     # log output
├── pruning_methods_classed.py  # core pruning methods
├── README.md
├── requirements.txt
├── src
│   ├── dataset_wrapper.py
│   ├── dora.py
│   ├── frequency_cnt.py
│   ├── peft_search_space.py
│   ├── pic_draw.py
│   ├── run_model.py
│   ├── search_turns.py
│   └── trainer_with_grad.py
├── submit                      # result files for glue benchmark
├── task_configs                # dataset and pruning method configs
│   ├── glue_full.yaml
│   ├── glue_full_base.yaml
│   └── glue_full_block.yaml
├── tools
│   └── controller.py           # entry point of PrunePEFT
└── utils
    ├── gpu_memory_plot.py
    ├── regression_head.py
    └── save_submit.py
```

## Run

- Run lora baseline on glue benchmark.

`python tools/controller.py --method method_configs/lora.yaml --task task_configs/glue_full_base.yaml --device 0`

- Run dynamic block prune on glue benchmark.

`python tools/controller.py --method method_configs/adapter_lora.yaml --task task_configs/glue_full_block.yaml --device 0`

- Run single pruning method on glue benchmark.

`python tools/controller.py --method method_configs/adapter_lora.yaml --task task_configs/glue_full.yaml --device 0`

device: the GPU id to use.
