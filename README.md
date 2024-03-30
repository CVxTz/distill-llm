# distill-llm

Example of distilling LLM knowledge using LoRa

## Data

We use this dataset ```juancavallotti/multilingual-gec``` from the huggingface Hub. It is a synthetic grammar correction dataset.

## Install

Torch

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Other libs:

```bash
pip install -r requirements.txt
```

## Run

All steps to run the experiments are listed in order in the file ```scripts/run.sh``` that you can run as ```bash scripts/run.sh```