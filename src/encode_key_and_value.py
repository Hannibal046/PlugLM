## 
##    --
##
"""
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12344 --use_env \
    encode_key_and_value.py \
        --ckpt_path \
        --config_path \
        --vocab_path \
        --tokenized_document_path \
        --output_dir \
        --return_value \
"""

import os
import numpy as np
import time
from run_pretraining import create_document_embedding
from modeling import RetrievalBertConfig,RetrievalBertForPreTraining
from utils import is_main_process, update_args,s2ms
import json
import torch
import argparse
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",type=str)
    parser.add_argument("--config_path",type=str)
    parser.add_argument("--vocab_path",type=str)
    parser.add_argument("--tokenized_document_path",type=str)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--return_value",action='store_true')
    parser.add_argument("--batch_size",type=int,default=8192)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if 'WORLD_SIZE' in os.environ else 0
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)
    config = json.load(open(args.config_path))
    model_args = config['model_args']
    model_args['tokenized_document_path'] = args.tokenized_document_path
    model_args = update_args(model_args,RetrievalBertConfig())
    model = RetrievalBertForPreTraining(model_args)
    missing_keys,unexpected_keys = model.load_state_dict(torch.load(args.ckpt_path),strict=False)
    if is_main_process():
        print("missing_keys",missing_keys)
        print("unexpected_keys",unexpected_keys)
    model = model.to(device)
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model,device_ids=[local_rank],output_device=local_rank)
    model = model.module if hasattr(model,'module') else model
    encoder  = model.get_document_encoder()
    tokenized_document = model.get_tokenized_document()
    if is_main_process():
        print("***** Running Encoding *****")
        print(f"  WorldSize = {world_size}")
        print(f"  Document = {args.tokenized_document_path}")
        print(f"  Model = {args.ckpt_path}")
        print(f"  Batch size per device = {args.batch_size}")
        print(f"  OutputDir = {args.output_dir}")
        print(f"  ReturnValue = {args.return_value}")
    start_time = time.time()
    os.makedirs(args.output_dir,exist_ok=True)
    create_document_embedding(encoder,tokenized_document,args.batch_size,output_dir=args.output_dir,return_value=args.return_value)
    end_time = time.time()
    if is_main_process():print(s2ms(end_time - start_time))