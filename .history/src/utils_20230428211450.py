"""
This file is used to store some utility functions organized as separate functions
"""
def set_available_port():
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    for port in range(12350,20000):
        if not port_is_used(port):
            os.environ['MASTER_PORT'] = str(port)
            break
    else:
        raise RuntimeError("No Available Port for DDP")

def wait_for_everyone():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        torch.distributed.barrier()
    else:
        return 


def postprocess_text(preds, labels):
    import nltk
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def get_rouge_score(hyps,refs):
    import rouge
    hyps,refs = postprocess_text(hyps,refs) # add \n

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            stemming=True)
    py_rouge_scores = evaluator.get_scores(hyps, refs)

    ## *100
    for k,v in py_rouge_scores.items():
        for _k,_v in v.items():
            py_rouge_scores[k][_k] = round(_v*100,4)

    return py_rouge_scores

def debpe(bpe):
    import re
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_bleu_score(hyps,refs,return_signature=False):
    """
    hyps:list of string
    refs:list of string
    """
    assert len(hyps) == len(refs)
    
    import sacrebleu
    scorer = sacrebleu.BLEU(force=True)
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score
    # return sacrebleu.corpus_bleu(hyps,[refs], 
    #                       force=True, lowercase=False,
    #                       tokenize='none').score
    # return sacrebleu.corpus_bleu(hyps,[refs], 
    #                       force=True, lowercase=False).score


def get_chrf_score(hyps,refs,return_signature=False):
    
    assert len(hyps) == len(refs)
    import sacrebleu
    scorer = sacrebleu.CHRF()
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score

def get_ter_score(hyps,refs,return_signature=False):
    
    assert len(hyps) == len(refs)
    import sacrebleu
    scorer = sacrebleu.TER()
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score

def port_is_used(port,ip='127.0.0.1'):
    """
    test whether a port is used or not
    """
    import socket
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.connect((ip,port))
        s.shutdown(2)
        return True
    except:
        return False

def save_config(data_args,model_args,training_args):
    from dataclasses import asdict
    import os
    import json
    ret = {
        "data_args":{},
        "model_args":{},
        "training_args":{},
    }
    for k,v in vars(data_args).items():ret["data_args"][k] = v
    for k,v in vars(model_args).items():ret["model_args"][k] = v
    for k,v in vars(training_args).items():ret["training_args"][k] = v

    with open(os.path.join(training_args.output_dir,'config.json'),'w') as f:
        json.dump(ret,f,indent=4)

def move_to_device(maybe_tensor, device):
    
    import torch
    import numpy as np

    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor


def update_args(args,model_args):
    from dataclasses import asdict
    import argparse
    """
    args:argparse.Namespace/dict
    # data_args:dataclass
    model_args:class
    # training_args:dataclass
    """
    model_keys = vars(model_args).keys()
    # print(model_keys)
    if isinstance(args,dict):
        for k,v in args.items():
            if k in model_keys:
                setattr(model_args,k,v)
    elif isinstance(args,argparse.Namespace):
        for key in vars(args):
            value = getattr(args,key)
            if value:
                # if key in model_keys:
                setattr(model_args,key,value)
    return model_args

def synchronize_args(data_args,model_args,training_args):
    # import os    
    data_args.train_batch_size = training_args.train_batch_size
    # data_args.eval_batch_size = training_args.eval_batch_size
    data_args.max_trg_len = model_args.max_trg_len
    data_args.max_src_len = model_args.max_src_len
    # if 'src.vocab' in os.listdir(data_args.data_path):
    #     ## separate vocab
    #     model_args.use_joint_bpe = False
    # else:
    #     model_args.use_joint_bpe = True
    return data_args,model_args,training_args

def get_remain_time(start,max_step,cur_step):
    import time
    end = time.time()
    past = end-start
    remain = (max_step/cur_step)*past - past
    return s2hm(remain)

def s2hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return  "%02d:%02d:%02d" % (h, m, s)
    # return  "%02d:%02d" % (h, m)

def s2hm(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return  "%02d:%02d" % (h, m)

def s2ms(s):
    m,s = divmod(s,60)
    return "%02d:%02d" % (m, s)

def get_rank():
    import torch.distributed as dist
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def safe_round(number, ndigits):
    import numpy as np
    import torch
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number

def get_perplexity(loss, round=2, base='e'):
    # from fairseq.logging.meters import safe_round
    import math

    if loss is None:
        return 0.0
    try:
        if base=='e':
            return safe_round(math.exp(loss), round)
        else:
            return safe_round(base**loss,round)
    except OverflowError:
        return float("inf")

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MetricsTracer:
    def __init__(self,valid_metrics):
        if valid_metrics == 'ppl':
            self.cur_best_metrics = 1000000
            self.better = "<"
        elif valid_metrics == 'bleu':
            self.cur_best_metrics = 0
            self.better = ">"
    def is_better(self,new_metrics):
        if eval(str(new_metrics)+self.better+str(self.cur_best_metrics)):
            self.cur_best_metrics = new_metrics
            return True
        else:
            return False

import torch
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class UnevenSequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    This is slightly different version of SequentialDistrbitedSample from 
    https://github.com/huggingface/transformers/blob/81ac45f85c35244831f11f73c09ea10eee4f953a/src/transformers/trainer_pt_utils.py
    In thie version, the datset is not evenly split. Since we don't need tensors of same shape to reduce or gather
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        import math
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas 
        indices = list(range(len(self.dataset)))
        self.indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples] ## a trick for python list ls[:infinity]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def dump_vocab(p,toker,direction='joint'):
    import os
    vocab = toker.dump_vocab() # dict {token:id}
    with open(os.path.join(p,direction+".vocab"),'w') as f:
        for k,v in vocab.items():
            f.write(k+'\t'+str(v)+'\n')

def get_current_gpu_usage():
    import GPUtil
    gpu = GPUtil.getGPUs()[0]
    return f"{gpu.memoryUsed}/{gpu.memoryTotal}"

import torch
class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self,documents) -> None:
        super().__init__()
        self.documents = documents
    def __len__(self):
        return len(self.documents)
    def __getitem__(self,idx):
        return self.documents[idx]

import torch
class ListDataset(torch.utils.data.Dataset):
    def __init__(self,ls) -> None:
        super().__init__()
        self.ls = ls
        self._len = len(self.ls)
    def __len__(self):
        return self._len
    def __getitem__(self,idx):
        return self.ls[idx]
    

def move_to_device(maybe_tensor, device):
    
    import torch
    import numpy as np

    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor

def debug_unsed_parameters(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)