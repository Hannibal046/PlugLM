# Decouple knowledge from parameters for plug-and-play language modeling

This reposotory contains the code and data for this ACL2023 Findings paper [Decouple knowledge from parameters for plug-and-play language modeling](https://arxiv.org/abs/2305.11564)

For the first time, we challenge the current implicit knowledge encoding mechanism for PLMs, which have two fundamental drawbacks: 

- The knowledge is neither editable nor scalable once the model is trained, which is especially problematic in that knowledge is consistently evolving. 
- It lacks interpretability and prevents humans from understanding which knowledge PLM requires for a certain problem. 

Based on the discovery that the Feed-Forward Network in PLM has the ability to store various types of knowledge during pre-training and is fundamentally a key-value memory network, we present **PlugLM**. This is the first model to separate knowledge storage from model parameters by utilizing an editable and scalable key-value memory, allowing for more adaptable and understandable knowledge encoding in PLMs.

We believe this architectural design would pave a new direction for future research on language model pre-training, especially for LLM.

<div align=center>
<img src=assets/model.svg width=90% height=90% />
</div>

## Setup
Our pretraining code is based on the [NVIDIA/BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT). 

First install the requirements listed there including:
- [PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/scripts/docker)
- [lddl](https://github.com/NVIDIA/lddl) is a utility library that minimizes the friction during dataset retrieval, preprocessing and loading for the language models in NVIDIA Deep Learning Examples.
- [APEX](https://github.com/NVIDIA/apex) is a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training, whereas AMP is an abbreviation used for automatic mixed precision training.
- [LAMB](https://arxiv.org/abs/1904.00962v1) stands for Layerwise Adaptive Moments based optimizer, is a large batch optimization technique that helps accelerate training of deep neural networks using large minibatches.

Then install the following packages:
```bash
pip install transformers tqdm numpy datasets
``` 

## Dataset
For pre-training data and pre-processing, we use [this scripts](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/data_download.sh) to download Wikipedia.

For S2ORC pre-training and fine-tuning data, we send application form and download from [here](https://github.com/allenai/s2orc).

For PubMed pre-training data, we download from [here](https://github.com/naver/biobert-pretrained), and Biomedical downstream dataset [here](https://github.com/dmis-lab/biobert).

For other datasets, we use [datasets](https://github.com/huggingface/datasets) to download.

## Pre-training
Here we provide the checkpoint of our model along with tokenized wikipedia knowledge base [here](https://drive.google.com/drive/folders/1tOznXhJ0ivEnvTmIaDvuiKyucb-8QPA6?usp=sharing).

The pre-training code (on 8*A100) is listed below:
```bash
cd src
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12344 --use_env run_pretraining.py \
    --data_dir ../data/phase2 \
    --output_dir ../results/pluglm \
    --tokenized_document_path ../data/tokenized_wiki.npy \
    --total_train_batch_size_per_device 8192 \
    --gradient_accumulation_steps 256 \
    --max_train_steps 8000 \
    --refreshing_steps 200 \
    --ckpt_steps 500 \
    --learning_rate=6e-3 \
    --refreshing_batch_size 4096 \
    --knowledge_attention_type multi_head
```
## Fine-tuning
Downstream finetuning code (for example GLUE/STS-B):
```bash
WKDIR=your/work/dir
RESULTDIR=$WKDIR/results/
PT=ckpt_8000.pt
MPI="-m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --use_env"
BS=batch_size
LR=learning_rate
MODEL=ckpt_path
TASK=stsb


python $MPI \
    $WKDIR/src/run_classification.py \
    --train_file $WKDIR/data/$TASK/train.jsonl \
    --dev_file $WKDIR/data/$TASK/$dev \
    --max_seq_len 128 \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --num_train_epochs 10 \
    --output_dir /tmp/ \
    --logging_steps -1 \
    --metrics $metrics \
    --fix_keys \
    --fix_values \
    --values_device gpu \
    --retrieval_bert \
    --ckpt_path $CKPTDIR/ckpt/$PT \
    --config_path $CKPTDIR/config.json \
    --vocab_path src/vocab/vocab \
    --keys_path $CKPTDIR/keys.npy \
    --values_path $CKPTDIR/values.npy
```

For other tasks, please refer to `run_ner.py`, `run_qa.py`, `run_classification.py`.

## Citation
If you find this project helpful, please consider cite our paper.
```
@article{cheng2023decouple,
  title={Decouple knowledge from paramters for plug-and-play language modeling},
  author={Cheng, Xin and Lin, Yankai and Chen, Xiuying and Zhao, Dongyan and Yan, Rui},
  journal={arXiv preprint arXiv:2305.11564},
  year={2023}
}
```
