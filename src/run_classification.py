## Built-in Module
import pickle
import os
import warnings
import numpy as np
import json
import math

# from retrieval_bert.modeling import BertForPreTraining
warnings.filterwarnings("ignore")
import time
from contextlib import nullcontext
from os import system as shell
from dataclasses import dataclass
from modeling import RetrievalBertConfig,RetrievalBertForSequenceClassification
from utils import update_args,wait_for_everyone
## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

## torch DDP
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

## huggingface/transformers
import transformers
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from transformers.tokenization_utils import PreTrainedTokenizerBase

## Third Party Module
import dllogger
from tqdm import tqdm


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())

## own
from utils import (
    ListDataset,
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
    move_to_device
)
from modeling import (
    RetrievalBertForPreTraining,
    RetrievalBertConfig,
    load_from_partial_bert,
)
from run_pretraining import (
    take_memory_refresh_step,
)
from metrics import (
    accuracy,
    binary_f1,
    micro_f1,
    macro_f1,
    spearman,
    pearson,
    matthews_corrcoef,
)

metric2func = {
    "accuracy":accuracy,
    "binary_f1":binary_f1,
    "macro_f1":macro_f1,
    "micro_f1":micro_f1,
    "pearson":pearson,
    "spearman":spearman,
    "matthews_corrcoef":matthews_corrcoef,
}

task2metrics = {
    'cola':matthews_corrcoef,
    'stsb':person_and_spearman,
    'mrpc':acc_and_f1,
    'qqp':acc_and_f1,
    'sst2':simple_accuracy,
    'mnli':simple_accuracy,
    'mnli_mismatched':simple_accuracy,
    'mnli_matched':simple_accuracy,
    'qnli':simple_accuracy,
    'rte':simple_accuracy,
    'wnli':simple_accuracy,
    'hans':simple_accuracy,
}



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## data_args
    data_args = parser.add_argument_group('data_args')
    data_args.add_argument('--train_file',type=str,default=None)
    data_args.add_argument('--dev_file',type=str,default=None)
    data_args.add_argument('--test_file',type=str,default=None)
    data_args.add_argument('--vocab_file',type=str,default='vocab/vocab')
    data_args.add_argument('--max_seq_len',default=128,type=int)
    data_args.add_argument('--is_regression',action='store_true')
    data_args.add_argument('--metrics',nargs="+",default=['accuracy'])
    data_args.add_argument("--tagged",action='store_true')

    ## model_args
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('--fix_keys',action='store_true')
    model_args.add_argument('--fix_values',action='store_true')
    model_args.add_argument('--retrieval_bert',action='store_true')
    model_args.add_argument('--ckpt_path',type=str)
    model_args.add_argument('--config_path',type=str)
    model_args.add_argument('--vocab_path',type=str)
    model_args.add_argument('--keys_path',nargs="+",)
    model_args.add_argument('--values_path',nargs="+",)
    model_args.add_argument('--values_device',type=str,default='cpu')
    model_args.add_argument('--return_I',action='store_true')
    
    ## training_args
    training_args = parser.add_argument_group('training_args')
    training_args.add_argument('--output_dir', type=str,default='/tmp/test')
    training_args.add_argument('--logging_steps', type=int,default=-1)
    # training_args.add_argument('--refreshing_steps', type=int,default=300)
    training_args.add_argument('--gradient_accumulation_steps', type=int,default=1)
    training_args.add_argument('--num_warmup_steps', type=int,default=0)
    training_args.add_argument('--learning_rate',type=float,default=5e-5)
    training_args.add_argument('--lr_scheduler_type', type=str,default='linear')
    training_args.add_argument('--per_device_train_batch_size', type=int,default=12)
    training_args.add_argument('--per_device_eval_batch_size', type=int,default=8)
    training_args.add_argument('--max_train_steps', type=int,default=None)
    training_args.add_argument('--num_train_epochs', type=int,default=3)
    training_args.add_argument('--max_grad_norm',type=float,default=1.0)
    training_args.add_argument('--weight_decay',type=float,default=0.0)
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

def flatten(container):
    def _flatten(container):
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i
    return list(_flatten(container))

def analysis_I(I,keys_len):
    ## cumsum
    boundary = []
    for idx,l in enumerate(keys_len):
        if idx == 0:
            boundary.append(l)
        else:
            boundary.append(boundary[idx-1]+l)
    # print(boundary)
    ## Turn all I into category
    for idx in range(len(I)):
        for jdx in range(len(I[idx])):
            for kdx in range(len(I[idx][jdx])):
                for ldx in range(len(boundary)):
                    if I[idx][jdx][kdx]<boundary[ldx]:
                        I[idx][jdx][kdx] = ldx
                        break
                assert I[idx][jdx][kdx] < len(boundary)
    # print(I)
    ## 总数占比
    from collections import Counter
    ls = flatten(I)
    print("总数占比:",Counter(ls))

    ## 有多少samples检索了相应的keys
    d = {}
    for idx in range(len(I)):
        s = set(flatten(I[idx]))
        for ele in s:
            d[ele] = d.get(ele,0)+1
    print("每个sample的检索情况:\n",d)
    
@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    pad_to_multiple_of:int = 8
    pad_token_id:int = 0

    def __call__(self, features):
        
        input_ids = [x['input_ids'] for x in features] 
        attention_mask = [x['attention_mask'] for x in features]
        labels = [x['label'] for x in features]
        
        max_seq_len = max(len(x) for x in input_ids)
        max_seq_len = max_seq_len + (self.pad_to_multiple_of - max_seq_len%self.pad_to_multiple_of)
        
        input_ids = [x+[self.pad_token_id]*(max_seq_len-len(x)) for x in input_ids]
        attention_mask = [x+[0]*(max_seq_len-len(x)) for x in attention_mask]

        pt = torch.tensor
        ret =  {
            'input_ids':pt(input_ids),
            'attention_mask':pt(attention_mask),
            'labels':pt(labels)
        }
        if "token_type_ids" in features[0].keys():
            token_type_ids = [x['token_type_ids'] for x in features]
            token_type_ids = [x+[self.pad_token_id]*(max_seq_len-len(x)) for x in token_type_ids]
            ret['token_type_ids'] = pt(token_type_ids)
        return ret

def get_I(dataloader,model):
    ## get model device
    device = next(model.parameters()).device

    def _get_I(dataloader,model,device):
        model.eval()
        I = []
        for batch in dataloader:
            batch = move_to_device(batch,device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            I.extend(outputs.I.cpu().tolist())
        return I

    if not dist.is_initialized():
        I = _get_I(dataloader,model,device)
        return I
    else:
        I = _get_I(dataloader,model,device)
        all_ranks = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_ranks,I)
        ## a list: [num_samples,num_layers,k]
        I = [x for y in all_ranks for x in y]
        torch.distributed.barrier()
        return I


def classify(dataloader,model,device,is_regression=False):
    def _classify(dataloader,model,device):
        model.eval()
        preds,labels = [],[]
        for batch in dataloader:
            batch = move_to_device(batch,device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                _preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                preds.append(_preds.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        return preds,labels
    
    if not dist.is_initialized():
        preds,labels = _classify(dataloader,model,device)
        return preds,labels
    else:
        preds,labels = _classify(dataloader,model,device)
        all_ranks = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_ranks,[preds,labels])
        preds = [x[0] for x in all_ranks]
        labels = [x[1] for x in all_ranks]
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        torch.distributed.barrier()
        return preds,labels
    
def prepare_logger(training_args):
    log_file = os.path.join(training_args.output_dir,"log.json")
    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,filename=log_file),
                            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    dllogger.metadata("avg_loss", {'format':":.3f"})
    dllogger.metadata("lr", {'format':":.9f"})
    dllogger.metadata("best_metrics", {'format':":.4f"})
    
def unify(dataset):
    
    def rm_keys(k):
        for d in dataset:
            d.pop(k)

    def ch_keys(old_key,new_key):
        for d in dataset:
            d[new_key] = d.pop(old_key)

    keys = dataset[0].keys()
    if set(keys) == set(["id","text","label"]):
        rm_keys('id')
    elif set(keys) == set(["id","text","label","headline"]):
        rm_keys('id')
        ch_keys('text','text2')
        ch_keys('headline','text1')
    elif set(keys) == set(["text","label","headline"]):
        ch_keys('text','text2')
        ch_keys('headline','text1')
    elif set(keys) == set(["metadata","text","label"]):
        rm_keys('metadata')
    elif set(keys) == set(["idx","sentence","label"]):
        rm_keys('idx')
        ch_keys('sentence','text')    
    elif set(keys) == set(["sentence","label"]):
        ch_keys('sentence','text')    
    elif set(keys) == set(["premise","hypothesis","label","idx"]):
        ch_keys('premise','text1')
        ch_keys('hypothesis','text2')
        rm_keys('idx')
    elif set(keys) == set(["sentence2","sentence1","label","idx"]):
        ch_keys('sentence1','text1')
        ch_keys('sentence2','text2')
    elif set(keys) == set(["question","sentence","label","idx"]):
        ch_keys('question','text1')
        ch_keys('sentence','text2')
        rm_keys('idx')
    elif set(keys) == set(["question1","question2","label","idx"]):
        ch_keys('question1','text1')
        ch_keys('question2','text2')
        rm_keys('idx')
    else:
        print(keys)
        raise RuntimeError
    
    return dataset

def tokenize_dataset(data,label2id,tokenizer,max_length,is_regression=False,tagged=False):
    
    
    if 'text' in data[0].keys():
        text = [x['text'] for x in data]
        if tagged:
            text = ["[unused1] " + x for x in text]
        features = tokenizer(text,padding=False,max_length=max_length,truncation=True)
    
    elif 'text1' in data[0].keys():
        text1 = [x['text1'] if x['text1'] is not None else "" for x in data]
        if tagged:
            text1 = ["[unused1] " + x for x in text1]
        text2 = [x['text2'] if x['text1'] is not None else "" for x in data]
        features = tokenizer(text1,text2,padding=False,max_length=max_length,truncation=True)

    labels = [x['label'] for x in data]
    if not is_regression:
        labels = [label2id[x] for x in labels]

    features['label']= labels
    ret = [dict() for _ in range(len(data))]
    
    for k,v in features.items():
        for idx,_v in enumerate(v):
            ret[idx][k] = _v
    
    return ret

def main(data_args=None,model_args=None,training_args=None):

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if 'WORLD_SIZE' in os.environ else 1
    is_ddp = world_size > 1

    ## ddp setup
    if is_ddp:
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)
    is_main = is_main_process()

    ## set seed
    set_seed(training_args.seed+local_rank)

    if is_main:
        ## back up source code,config,tokenizer
        os.makedirs(training_args.output_dir,exist_ok=True)
        # os.makedirs(os.path.join(training_args.output_dir,"ckpt"),exist_ok=True)
        # shell(f"rsync -a {os.getcwd()} {training_args.output_dir} --exclude dllogger --exclude lamb_amp_opt --exclude hgbert --exclude lddl")
        # save_config(data_args,model_args,training_args)
        prepare_logger(training_args)
    else:
        dllogger.init(backends=[])

    train_data = [json.loads(x) for x in open(data_args.train_file).readlines()]
    dev_data = [json.loads(x) for x in open(data_args.dev_file).readlines()]
    if data_args.test_file is not None:
       test_data =  [json.loads(x) for x in open(data_args.test_file).readlines()]
    
    ## 不同数据集之间的适配,改变k
    train_data = unify(train_data)
    dev_data = unify(dev_data)
    if data_args.test_file is not None: test_data = unify(test_data)

    labels = list(set([x['label'] for x in train_data]))
    labels.sort()
    if not data_args.is_regression:
        num_labels = len(labels)
    else:
        num_labels = 1
    label2id = {label:idx for idx,label in enumerate(labels)}
    id2label = {idx:label for idx,label in enumerate(labels)}

    pretrain_config = None
    if not model_args.retrieval_bert:
        config = AutoConfig.from_pretrained(model_args.ckpt_path,num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_args.ckpt_path,config=config).to(device)
        toker = AutoTokenizer.from_pretrained(model_args.ckpt_path)
    else:
        config = json.load(open(model_args.config_path))
        pretrain_config = config['model_args']
        pretrain_config['fix_keys'] = model_args.fix_keys
        pretrain_config['fix_values'] = model_args.fix_values
        pretrain_config['num_labels'] = num_labels
        if model_args.fix_keys and model_args.fix_values:
            pretrain_config['tokenized_document_path'] = None
        pretrain_config["return_I"] = model_args.return_I
        pretrain_config = update_args(pretrain_config,RetrievalBertConfig())
        pretrain_config.num_labels = num_labels
        model = RetrievalBertForSequenceClassification(pretrain_config)
        pretrained_state_dict = torch.load(model_args.ckpt_path,map_location='cpu')
        missing_keys,unexpected_keys=model.load_state_dict(pretrained_state_dict,strict=False)
        # if is_main:
            # print("pretrained_keys",pretrained_state_dict.keys())
            # print("missing_keys",missing_keys)
            # print("unexpected_keys",unexpected_keys)
        model = model.to(device)

        keys = []
        keys_len = []
        for k in model_args.keys_path:
            _k = np.load(k)
            keys.append(_k)
            keys_len.append(_k.shape[0])
            # if is_main_process():
            #     print(f"loading keys from {k}")
            #     print(f"shape:{keys[-1].shape}")
        keys = np.concatenate(keys)
        # keys = np.load(model_args.keys_path)
        values = None
        if model_args.fix_values:
            values = []
            for v in model_args.values_path:
                values.append(np.load(v))
            values = np.concatenate(values)
            # values = np.load(model_args.values_path)
        _device = device if model_args.values_device == 'gpu' else 'cpu'
        # if is_main:
        #     print("values device",_device)
        #     print("values.shape",values.shape)
        #     print("keys.shape",keys.shape)
        model.update_memory(keys,values,_device)
        # if is_main:shell("nvidia-smi")

        toker = transformers.BertTokenizerFast(model_args.vocab_path)
        if data_args.tagged:
            toker.add_special_tokens({ "additional_special_tokens": [ "[unused1]" ] })
    
    collator = DataCollatorWithPadding(toker,pad_token_id=toker.pad_token_id)
    train_dataset = ListDataset(tokenize_dataset(train_data,label2id,toker,data_args.max_seq_len,is_regression=data_args.is_regression,tagged=data_args.tagged))
    dev_dataset = ListDataset(tokenize_dataset(dev_data,label2id,toker,data_args.max_seq_len,is_regression=data_args.is_regression,tagged=data_args.tagged))
    if data_args.test_file:
        test_dataset = ListDataset(tokenize_dataset(test_data,label2id,toker,data_args.max_seq_len,is_regression=data_args.is_regression,tagged=data_args.tagged))
    
    if is_ddp:

        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,collate_fn=collator,shuffle=False,
                                      batch_size = training_args.per_device_train_batch_size,sampler = train_sampler,)

        dev_sampler = UnevenSequentialDistributedSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset,collate_fn = collator,batch_size = training_args.per_device_eval_batch_size,
                                    shuffle=False,sampler=dev_sampler)
        if data_args.test_file:
            test_sampler = UnevenSequentialDistributedSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset,collate_fn=collator,batch_size = training_args.per_device_eval_batch_size,
                                        shuffle=False,sampler=test_sampler)
    else:
        train_dataloader = DataLoader(train_dataset,collate_fn=collator,shuffle=True,batch_size = training_args.per_device_train_batch_size,
                                    num_workers=0,pin_memory=False,)

        dev_dataloader = DataLoader(dev_dataset,collate_fn=collator,batch_size = training_args.per_device_eval_batch_size,
                                    num_workers=0,pin_memory=False,shuffle=False)
        if data_args.test_file:
            test_dataloader = DataLoader(test_dataset,collate_fn=collator,batch_size = training_args.per_device_eval_batch_size,
                                        num_workers=0,pin_memory=False,shuffle=False,)  
    
    train_iter = tqdm(
        train_dataloader,
        desc="Iteration",
        disable=not training_args.enable_progress_bar,
        total=len(train_dataloader),
    ) if is_main else train_dataloader
    
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model,device_ids=[local_rank],output_device=local_rank)#,find_unused_parameters=True)
    model.train()

    ## prepare optimizer and lr_scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    ## start training
    completed_steps = 0
    scaler = torch.cuda.amp.GradScaler()
    epoch = 1
    start_time = time.time()
    average_loss = torch.tensor([0.0]).to(device)
    num_samples = torch.tensor([0]).to(device)
    skip_optimization_steps = 0
    best_metrics = -1
    ckpt_path = []
    ## Training
    # if is_main:
    #     print("***** Running training *****")
    #     print(f"  OutputDir = {training_args.output_dir}")
    #     print(f"  Dataset = {data_args.train_file}")
    #     print(f"  TrainData = {len(train_dataset)}")
    #     print(f"  DevData = {len(dev_dataset)}")
    #     if data_args.test_file:
    #         print(f"  TestData = {len(test_dataset)}")
    #     print(f"  Model = {model_args.ckpt_path}")
    #     print(f"  Total Batch size = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")
    #     print(f"  Batch Size Per Device = {training_args.per_device_train_batch_size}")
    #     # print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    #     print(f"  World Size = {world_size}")
    #     print(f"  Lr = {training_args.learning_rate}")
    #     # print(f"  LrSchedulerType = {training_args.lr_scheduler_type}")
    #     # print(f"  Total optimization steps = {training_args.max_train_steps}")
    print(f"  Model Params = {sum([x.numel() for x in model.parameters() if x.requires_grad])/1e6:.2f}M")

    for epoch in range(1,training_args.num_train_epochs+1):
        model.train()
        epoch_start = time.time()
        for step,batch in enumerate(train_iter,start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
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
                if training_args.max_grad_norm>0:
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
                if completed_steps%training_args.logging_steps==0 and completed_steps>0 and training_args.logging_steps != -1:
                    # if is_ddp:
                    #     dist.all_reduce(average_loss,op=dist.ReduceOp.SUM)
                    #     dist.all_reduce(num_samples,op=dist.ReduceOp.SUM)
                    if is_main:
                        dllogger.log(step=(epoch, completed_steps,),
                                    data={"avg_loss": (average_loss / num_samples).item(),
                                        "best_metrics":best_metrics,
                                        # "lr": (optimizer.param_groups[0]['lr']),
                                        "skip_optimization_steps":skip_optimization_steps,
                                        # "elapsed":s2hm(time.time()-start_time),
                                        "remaining":get_remain_time(start_time,training_args.max_train_steps,completed_steps)
                                        }
                                    )
                    average_loss.zero_()   
                    num_samples.zero_()
        epoch_end = time.time()
        start_time = time.time()
        preds,labels = classify(dev_dataloader,model,device,is_regression=data_args.is_regression)
        end_time = time.time()
        results = {"epoch":epoch,"inference_time":end_time-start_time,"training_time":s2ms(epoch_end-epoch_start)}
        for metric in data_args.metrics:results['dev_'+metric] = metric2func[metric](preds,labels)
        # if is_main:print(results)
        if results['dev_'+data_args.metrics[0]]>best_metrics:
            best_epoch = epoch
            best_metrics = results['dev_'+data_args.metrics[0]]
            model_to_save = model.module if hasattr(model,'module') else model
            if is_main:torch.save(model_to_save.state_dict(),os.path.join(training_args.output_dir,'best_model.ckpt'))

    wait_for_everyone()

    results = {}
    results['best_epoch'] = best_epoch
    results['dev_'+data_args.metrics[0]] = best_metrics
    
    
    ## evaluate model on test set
    if data_args.test_file:
        ckpt_path = os.path.join(training_args.output_dir,'best_model.ckpt')
        # if is_main:print("best_model_path:",ckpt_path)
        if is_ddp:
            model.module.load_state_dict(torch.load(ckpt_path,map_location=device))
            wait_for_everyone()
        else:
            model.load_state_dict(torch.load(ckpt_path,map_location=device))
        
        ## classification
        test_start_time = time.time()
        preds,labels = classify(test_dataloader,model,device,is_regression=data_args.is_regression)
        results['test_time'] = time.time() - test_start_time
        print(results['test_time'])
        
        ## get I
        I = None
        if pretrain_config is not None and pretrain_config.return_I:
            I = get_I(test_dataloader,model)
            if is_main:
                with open(os.path.join(training_args.output_dir,"I.pkl"),'wb') as f:
                    pickle.dump(I,f)
                print('\n'.join(model_args.keys_path))
                print(keys_len)
                analysis_I(I,keys_len)
        for metric in data_args.metrics:
            results['test_'+metric] = metric2func[metric](preds,labels)

    if is_main:
        print(json.dumps(results))

    dllogger.flush()

if __name__ == '__main__':
    
    args_group = parse_args()
    data_args,model_args,training_args  = args_group['data_args'],args_group['model_args'],args_group['training_args']
    # model_args = update_args(args=model_args,model_args = RetrievalBertConfig())
    # print("data_args:",data_args,"\nmodel_args:",model_args,"\ntraining_args:",training_args)
    main(data_args=data_args,model_args=model_args,training_args=training_args)

