## Built-in Module
import os
import warnings
import numpy as np

# from retrieval_bert.modeling import BertForPreTraining
warnings.filterwarnings("ignore")
import time
from contextlib import nullcontext
from os import system as shell

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F

## torch DDP
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

## huggingface/transformers
from transformers import set_seed

## Third Party Module
import dllogger
import logging
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from tqdm import tqdm
import lddl.torch
## own
from utils import (
    UnevenSequentialDistributedSampler,
    is_main_process,
    save_config,
    update_args,
    get_remain_time,
    s2hm,
    format_step,
    get_current_gpu_usage,
    get_rank,
    s2ms,
    DocumentDataset,
    wait_for_everyone,
    debug_unsed_parameters,
)
from modeling import (
    RetrievalBertForPreTraining,
    RetrievalBertConfig,
    load_from_partial_bert,
    RetrievalBertForMaskedLM,
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## data_args
    data_args = parser.add_argument_group('data_args')
    data_args.add_argument('--data_dir',type=str,required=True)
    data_args.add_argument('--vocab_file',type=str,default='vocab/vocab')
    
    ## model_args
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('--num_hidden_layers',type=int,default=12)
    model_args.add_argument('--hidden_size', type=int,default=768)
    model_args.add_argument('--vocab_size', type=int,default=30522)
    model_args.add_argument('--intermediate_size', type=int,default=3072)
    model_args.add_argument('--hidden_dropout_prob', type=float,default=0.1)
    model_args.add_argument('--num_attention_heads', type=int,default=12)
    model_args.add_argument('--hidden_act', type=str,default='gelu')
    model_args.add_argument('--init_from_bert',action='store_true')
    model_args.add_argument('--pretrained_bert_path', type=str,default='../pretrained_model/bert_base_uncased/pytorch_model.bin')
    model_args.add_argument('--memory_k',default=5,type=int)
    model_args.add_argument('--query_size',default=128,type=int)
    model_args.add_argument('--memory_layer_ffn_dim',default=768,type=int)
    model_args.add_argument('--num_memory_layers',default=1,type=int)
    model_args.add_argument('--tokenized_document_path',default='../data/tokenized_wiki.npy',type=str)
    model_args.add_argument('--fix_keys',action='store_true')
    model_args.add_argument('--fix_values',action='store_true')
    model_args.add_argument('--knowledge_attention_type',default='multi_head') # [sentence_level,multi_head]
    model_args.add_argument('--document_encoder_type',default='word_embedding')
    model_args.add_argument('--document_encoder_pooler_type',default='attentive')
    model_args.add_argument('--memory_pooler_type',default='attentive')
    
    
    
    ## training_args
    training_args = parser.add_argument_group('training_args')
    training_args.add_argument('--output_dir', type=str,required=True)
    training_args.add_argument('--logging_steps', type=int,default=1)
    training_args.add_argument('--refreshing_steps', type=int,default=300)
    training_args.add_argument('--ckpt_steps', type=int,default=500)
    training_args.add_argument('--max_ckpt', type=int,default=5)
    training_args.add_argument('--gradient_accumulation_steps', type=int,default=512)
    training_args.add_argument('--warmup_proportion', type=float,default=0.2843)
    training_args.add_argument('--learning_rate',type=float,default=4e-3)
    training_args.add_argument('--total_train_batch_size_per_device', type=int,default=8192) ## to keep align with NvBert, this is total batch size
    training_args.add_argument('--train_batch_size_per_device', type=int,default=0)
    training_args.add_argument('--max_train_steps', type=int,default=10_0000)
    training_args.add_argument('--max_grad_norm',type=float,default=1.0)
    training_args.add_argument('--weight_decay',type=float,default=0.01)
    training_args.add_argument('--seed',type=int,default=42)
    training_args.add_argument('--enable_progress_bar',action='store_true')
    training_args.add_argument('--num_workers',type=int,default=4)
    training_args.add_argument('--refreshing_batch_size',type=int,default=48)
    args = parser.parse_args()

    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)
    return arg_groups
    
def create_document_embedding(encoder,documents,batch_size,output_dir,return_value,enable_progress_bar=False):

    def collate_fn(batch):
        input_ids,attention_mask = [],[]
        for sample in batch:
            input_ids.append(sample)
            attention_mask.append(torch.tensor([int(x) for x in sample!=0]))
        return torch.stack(input_ids),torch.stack(attention_mask)

    rank = get_rank()
    encoder.eval()
    dataset = DocumentDataset(documents)
    if not dist.is_initialized():
        dataloader = torch.utils.data.DataLoader(dataset,batch_size,shuffle=False,collate_fn=collate_fn)
    else:
        sampler = UnevenSequentialDistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size,sampler=sampler,collate_fn=collate_fn)
    with torch.no_grad():
        document_keys = []
        document_values = []
        cnt = 0

        dataloader = tqdm(
            dataloader,
            desc="Iteration",
            disable=not enable_progress_bar,
            total=len(dataloader),
            ) if is_main_process() else dataloader

        for idx,(input_ids,attention_mask) in enumerate(dataloader):
            input_ids,attention_mask = input_ids.to(rank).to(dtype=torch.int32),attention_mask.to(rank)
            if return_value:
                key,value = encoder.encode(input_ids,attention_mask,return_value=True)
                document_values.append(value.cpu().numpy())
            else:
                key = encoder.encode(input_ids,attention_mask)
            document_keys.append(key.cpu().numpy())
            # document_keys.append(key)
            cnt += input_ids.shape[0]
    
    np.save(os.path.join(output_dir,'temp.key.'+str(rank)+'.npy'),np.concatenate(document_keys))
    if return_value: np.save(os.path.join(output_dir,'temp.value.'+str(rank)+'.npy'),np.concatenate(document_values))
    wait_for_everyone()

    if rank==0:
        keys = []
        key_ls = [os.path.join(output_dir,x) for x in os.listdir(output_dir) if x.startswith('temp.key') and x.endswith('npy')]
        key_ls.sort(key=lambda x:int(x.split(".")[-2]))
        print(key_ls)
        for k in key_ls:keys.append(np.load(k))
        keys = (np.concatenate(keys)).astype("float16")
        np.save(os.path.join(output_dir,"keys.npy"),keys)

        if return_value:
            values = []
            value_ls = [os.path.join(output_dir,x) for x in os.listdir(output_dir) if x.startswith('temp.value') and x.endswith('npy')]
            value_ls.sort(key=lambda x:int(x.split(".")[-2]))
            for v in value_ls:values.append(np.load(v))
            values = (np.concatenate(values)).astype("float16")
            np.save(os.path.join(output_dir,"values.npy"),values)
        
    encoder.train()
    if dist.is_initialized():torch.distributed.barrier()
    os.remove(os.path.join(output_dir,'temp.key.'+str(rank)+'.npy')) 
    if return_value:os.remove(os.path.join(output_dir,'temp.value.'+str(rank)+'.npy'))
    return            

def take_memory_refresh_step(model,batch_size = 2048,output_dir=None,return_value=False,update=True):
    
    model = model.module if hasattr(model,'module') else model
    encoder  = model.get_document_encoder()
    tokenized_document = model.get_tokenized_document()
    s = time.time()
    create_document_embedding(encoder,tokenized_document,batch_size,output_dir=output_dir,return_value=return_value)
    re_encoding_time = time.time()-s    
    if is_main_process():print("Re_encoding_time:",s2ms(re_encoding_time))
    if update:
        keys = np.load(os.path.join(output_dir,'keys.npy'))
        values=None
        if return_value:values = np.load(os.path.join(output_dir,'values.npy'))
        model.update_memory(keys,values)
        wait_for_everyone()
    return 
    
def prepare_optimization(training_args,model):
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=training_args.learning_rate,
                             max_grad_norm = training_args.max_grad_norm)
    
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters,lr = training_args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=training_args.warmup_proportion,
                                       total_steps=training_args.max_train_steps,
                                       base_lr=training_args.learning_rate,
                                       device=model.device)
    optimizer.setup_fp32_params()
    return optimizer,lr_scheduler

def prepare_logger(training_args):
    log_file = os.path.join(training_args.output_dir,"log.json")
    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,filename=log_file),
                            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    # dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,filename=log_file)])
    dllogger.metadata("avg_loss", {'format':":.3f"})
    dllogger.metadata("lr", {'format':":.9f"})

def main(data_args=None,model_args=None,training_args=None):

    
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if 'WORLD_SIZE' in os.environ else 0
    is_ddp = world_size > 1

    ## ddp setup
    if is_ddp:
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)
    is_main = is_main_process()

    ## set seed
    set_seed(training_args.seed+local_rank)

    if training_args.total_train_batch_size_per_device % training_args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            training_args.gradient_accumulation_steps, training_args.total_train_batch_size_per_device))

    training_args.train_batch_size_per_device = training_args.total_train_batch_size_per_device // training_args.gradient_accumulation_steps

    if is_main:
        ## back up source code,config,tokenizer
        os.makedirs(training_args.output_dir,exist_ok=True)
        os.makedirs(os.path.join(training_args.output_dir,"ckpt"),exist_ok=True)
        shell(f"rsync -a {os.getcwd()} {training_args.output_dir} --exclude dllogger --exclude lamb_amp_opt --exclude hgbert --exclude lddl")
        save_config(data_args,model_args,training_args)
        prepare_logger(training_args)
    else:
        dllogger.init(backends=[])

    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
        data_args.data_dir,
        local_rank=max(local_rank, 0),
        vocab_file=data_args.vocab_file,
        data_loader_kwargs={
            'batch_size': training_args.train_batch_size_per_device,
            'num_workers': training_args.num_workers,
            'pin_memory': True,
        },
        base_seed=training_args.seed,
        log_dir=None if training_args.output_dir is None else os.path.join(training_args.output_dir, 'lddl_log'),
        log_level=logging.WARNING,
        start_epoch=0,
    )

    train_iter = tqdm(
        train_dataloader,
        desc="Iteration",
        disable=not training_args.enable_progress_bar,
        total=len(train_dataloader),
    ) if is_main else train_dataloader
    
    ## load model
    # model = BertForPreTraining(model_args).to(device)
    # model = RetrievalBertForPreTraining(model_args).to(device)
    model = RetrievalBertForMaskedLM(model_args).to(device)

    if model_args.init_from_bert:
        model = load_from_partial_bert(model,model_args.pretrained_bert_path)
    # if is_main:print(model)
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model,device_ids=[local_rank],output_device=local_rank)#,find_unused_parameters=True)
    model.train()

    take_memory_refresh_step(model,training_args.refreshing_batch_size,training_args.output_dir)
    
    ## load optimizer and lr_scheduler
    optimizer,lr_scheduler = prepare_optimization(training_args,model)
    
    ## start training
    completed_steps = 0
    scaler = torch.cuda.amp.GradScaler()
    epoch = 1
    start_time = time.time()
    average_loss = torch.tensor([0.0]).to(device)
    num_samples = torch.tensor([0]).to(device)
    skip_optimization_steps = 0
    ckpt_path = []
    ## Training
    if is_main:
        print("***** Running training *****")
        print(f"  Total Batch size per device = {training_args.total_train_batch_size_per_device}")
        print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        print(f"  Real Batch Size = {training_args.total_train_batch_size_per_device//training_args.gradient_accumulation_steps}")
        print(f"  Total Batch Size = {training_args.total_train_batch_size_per_device*world_size}")
        print(f"  World Size = {world_size}")
        print(f"  PeakLR = {training_args.learning_rate}")
        print(f"  Total optimization steps = {training_args.max_train_steps}")
        print(f"  Warmup_proportion = {training_args.warmup_proportion}")
        print(f"  Vocab Size = {model_args.vocab_size}")
        print(f"  Model Params = {sum([x.numel() for x in model.parameters() if x.requires_grad])/1e6:.2f}M")
    ## max_step training
    while completed_steps <= training_args.max_train_steps:
        for step,batch in enumerate(train_iter,start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k !="next_sentence_labels"}
            maybe_post_communicate = model.no_sync if is_ddp and step % training_args.gradient_accumulation_steps != 0 else nullcontext
            with maybe_post_communicate():
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                loss = outputs.loss 
                bs = batch['input_ids'].shape[0]
                average_loss += (loss.clone().detach()*bs)
                num_samples += bs
                loss = loss / training_args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            ## update parameters
            if step%training_args.gradient_accumulation_steps==0 or step == len(train_dataloader):    
                if training_args.max_grad_norm>0 and False: ## LAMB optimizer have already take care of gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                scaler.step(optimizer)
                _scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skip_lr_sched = (_scale > scaler.get_scale())
                skip_optimization_steps += int(skip_lr_sched)
                if not skip_lr_sched:
                    completed_steps += 1
                    lr_scheduler.step()    
                ## logging when training
                if completed_steps%training_args.logging_steps==0 and completed_steps>0:
                    # if is_ddp:
                    #     dist.all_reduce(average_loss,op=dist.ReduceOp.SUM)
                    #     dist.all_reduce(num_samples,op=dist.ReduceOp.SUM)
                    if is_main:
                        dllogger.log(step=(epoch, completed_steps,),
                                    data={"avg_loss": (average_loss / num_samples).item(),
                                        "lr": (optimizer.param_groups[0]['lr']).item(),
                                        "skip_optimization_steps":skip_optimization_steps,
                                        "gpu":get_current_gpu_usage(),
                                        "elapsed":s2hm(time.time()-start_time),
                                        "remaining":get_remain_time(start_time,training_args.max_train_steps,completed_steps)
                                        }
                                    )
                    average_loss.zero_()   
                    num_samples.zero_()

                ## saving ckpt
                if completed_steps%training_args.ckpt_steps==0 and completed_steps > 0:
                    if is_main:
                        model_to_be_saved = model.module if hasattr(model,'module') else model
                        if len(ckpt_path)==training_args.max_ckpt:
                            delete_path = ckpt_path.pop(0)
                            shell(f"rm -rf {delete_path}")
                        save_path = os.path.join(training_args.output_dir,"ckpt",f"ckpt_{completed_steps}.pt")
                        ckpt_path.append(save_path)
                        torch.save(model_to_be_saved.state_dict(),save_path)

                ## memory refreshing
                if completed_steps % training_args.refreshing_steps == 0 and completed_steps > 0:
                    take_memory_refresh_step(model,training_args.refreshing_batch_size,training_args.output_dir)

                if completed_steps > training_args.max_train_steps:break

        epoch += 1  

    take_memory_refresh_step(model,training_args.refreshing_batch_size,training_args.output_dir,return_value=True,update=False)
    dllogger.flush()

if __name__ == '__main__':
    
    args_group = parse_args()
    data_args,model_args,training_args  = args_group['data_args'],args_group['model_args'],args_group['training_args']
    model_args = update_args(args=model_args,model_args = RetrievalBertConfig())
    main(data_args=data_args,model_args=model_args,training_args=training_args)

