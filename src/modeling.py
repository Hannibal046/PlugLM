import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import *
from transformers.models.bert.configuration_bert import *
# from transformers.modeling_bert import *
# from transformers.configuration_bert import *
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss,BCELoss,BCEWithLogitsLoss

@dataclass
class RetrieverOutput:
    key:torch.Tensor=None
    value:torch.Tensor=None
    I:torch.Tensor=None

@dataclass
class MemoryLayerOutput:
    hidden_states:torch.Tensor=None
    I:torch.Tensor=None

@dataclass
class RetrievalBertLayerOutput(BaseModelOutputWithPastAndCrossAttentions):
    I:torch.Tensor=None

@dataclass
class RetrievalBertModelOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    I:torch.Tensor=None

@dataclass
class RetrievalSequenceClassifierOutput(SequenceClassifierOutput):
    I:torch.Tensor=None

def get_rank():
    import torch.distributed as dist
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_current_gpu_usage():
    import GPUtil
    gpu = GPUtil.getGPUs()[0]
    return f"{gpu.memoryUsed}/{gpu.memoryTotal}"

def load_from_partial_bert(model,pretrained_path):
    pretrained_dict = dict(torch.load(pretrained_path))
    
    ## revised gamma and beta for layer norm
    pretrained_keys = list(pretrained_dict.keys())
    for k in pretrained_keys:
        if 'LayerNorm.gamma' in k:
            new_k = k.replace('gamma','weight')
            pretrained_dict[new_k] = pretrained_dict.pop(k)
        elif 'LayerNorm.beta' in k:
            new_k = k.replace('beta','bias')
            pretrained_dict[new_k] = pretrained_dict.pop(k)
    pretrained_keys = list(pretrained_dict.keys())
    model_dict = model.state_dict()
    for k in model_dict:
        if k in pretrained_dict:
            model_dict[k] = pretrained_dict[k]
            pretrained_keys.remove(k)
    model.load_state_dict(model_dict)
    
    if is_main_process(): print("The following keys in Pretrained Bert are not used:",pretrained_keys)
    return model

class RetrievalBertConfig(PretrainedConfig):
    model_type = "retrieval_bert"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_labels=2,
        ## relevant to retrieval bert
        memory_k = 5,
        query_size = 128,
        tokenized_document_path = None,
        fix_keys = False,
        fix_values = False,
        knowledge_attention_type = 'multi_head',
        memory_pooler_type = 'attentive',
        query_batchnorm = True,
        memory_layer_ffn_dim = 768,
        num_memory_layers = 6,
        init_from_bert = False,
        pretrained_bert_path = None,
        document_encoder_type = 'word_embedding',
        document_encoder_pooler_type = 'attentive',
        return_I = False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
 
        self.return_I = return_I
        self.num_labels = num_labels
        self.document_encoder_pooler_type = document_encoder_pooler_type
        self.document_encoder_type = document_encoder_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.memory_k = memory_k
        self.query_size = query_size
        self.tokenized_document_path = tokenized_document_path
        self.fix_keys = fix_keys
        self.fix_values = fix_values
        self.knowledge_attention_type = knowledge_attention_type
        self.memory_pooler_type = memory_pooler_type
        self.query_batchnorm = query_batchnorm
        self.memory_layer_ffn_dim = memory_layer_ffn_dim
        self.num_memory_layers = num_memory_layers
        self.init_from_bert = init_from_bert
        self.pretrained_bert_path = pretrained_bert_path

class Pooler(nn.Module):
    def __init__(self,pooler_type,d_model=None) -> None:
        super().__init__()
        if pooler_type == 'avg':
            self.pooler = AveragePooler()
        elif pooler_type == 'attentive':
            assert d_model is not None
            self.pooler = AttentivePooler(d_model)
        elif pooler_type == 'max':
            self.pooler = MaxPooler()
        elif pooler_type == 'cls':
            self.pooler = ClsPooler()
    def forward(
        self,
        hidden_states, # [bs,seq_len,d_model]
        attention_mask = None, 
    ):
        if attention_mask is not None and attention_mask.ndim >= 3:
            attention_mask = (1 - (attention_mask.squeeze(1).squeeze(1) / -10000.0))
        ## hidden_states: bs,seq_len,d_model
        ## attention_mask: bs,seq_len [1,1,1,0,0,0]  
        return self.pooler(hidden_states,attention_mask)
        
class AttentivePooler(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.att_fc1 = nn.Linear(d_model,d_model)
        self.att_fc2 = nn.Linear(d_model,1)
        self.apply(self.init_weights)
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def forward(
        self,
        x, 
        attn_mask = None 
    ):  
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x  

class Retriever(nn.Module):
    def __init__(self,config,embedding,bert_layer) -> None:
        super().__init__()
        
        self.keys = None
        self.values = None
        self.k = config.memory_k
        self.pad_token_id = config.pad_token_id
        self.config = config

        if not config.fix_keys:
            self.key_proj = nn.Linear(config.hidden_size,config.query_size)
        if not config.fix_values:
            self.value_proj = nn.Linear(config.hidden_size,config.hidden_size)
        if False in (self.config.fix_keys,self.config.fix_values):
            self.document_encoder = DocumentEncoder(config,embedding,bert_layer)
        self.tokenized_documents = None
        if config.tokenized_document_path:
            self.tokenized_documents = torch.tensor(np.load(config.tokenized_document_path),dtype=torch.int16)    

    def forward(
        self,
        queries, #[bs,query_size]
    ):  
        ## topk retrieval
        bs,_ = queries.shape
        gpu_device = queries.device

        _,I = torch.topk(queries@self.keys.permute(1,0),self.k)

        I = I.view(-1)
        if self.tokenized_documents is not None:
            I = I.to(self.tokenized_documents.device)
            
        ## TODO maybe add some noise here

        if False in (self.config.fix_keys,self.config.fix_values):
            ## prepare document input_ids and attention_mask
            tokenized_documents = self.tokenized_documents.index_select(0,I).view(bs*self.k,-1).to(gpu_device).to(dtype=torch.int32) # bs,topk,seq_len
            attention_mask = (tokenized_documents != self.pad_token_id).int().to(gpu_device).to(dtype=self.value_proj.weight.dtype)
            document_embedding = self.document_encoder(tokenized_documents,attention_mask)
            ## get key
            if not self.config.fix_keys:
                key = self.key_proj(document_embedding).view(bs,self.k,-1)
            else:
                key = self.keys.index_select(0,I.to(self.keys.device)).view(bs,self.k,-1)
            ## get value
            if not self.config.fix_values:
                value = self.value_proj(document_embedding).view(bs,self.k,-1)
            else:
                value = self.values.index_select(0,I.to(self.values.device)).view(bs,self.k,-1)
                value = value.to(key.device)
        else:
            key = self.keys.index_select(0,I).view(bs,self.k,-1)
            value = self.values.index_select(0,I.to(self.values.device)).view(bs,self.k,-1).to(key.device)
        # return key,value,I.view(bs,self.k)
        if self.config.return_I:
            I = I.view(bs,-1).to('cpu')
        else:
            I = None
        return RetrieverOutput(
            key=key,
            value=value,
            I=I
        )
    
    def encode(self,input_ids,attention_mask,return_value = False):
        document_embedding = self.document_encoder(input_ids,attention_mask)
        if return_value:
            return self.key_proj(document_embedding),self.value_proj(document_embedding)
        else:
            return self.key_proj(document_embedding)

class BiasedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.0,
        bias = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim //num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.beta = nn.Parameter(torch.ones(num_heads),requires_grad=True) ## [num_heads]

    def _shape(self,tensor,seq_len,bsz):
        return tensor.view(bsz,seq_len,self.num_heads,self.head_dim).transpose(1,2).contiguous()
    
    def forward(
        self,
        hidden_states, # [bs,tgt_len,d_model]
        key_value_states, # [bs,src_len,d_model]
        bias, # bs,num_heads,tgt_len,src_len
    ):
        bsz,tgt_len,_ = hidden_states.size()
        query_states = self.q_proj(hidden_states)*self.scaling ## bs,tgt_len,d_model
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz) ## bs,num_head,k,head_dim
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz) ## bs,num_head,k,head_dim
        proj_shape = (bsz * self.num_heads, -1, self.head_dim) 
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # bsz*num_head,tgt_len,head_dim
        key_states = key_states.view(*proj_shape) # bsz*num_head,src_len,head_dim
        value_states = value_states.view(*proj_shape) # bsz*num_head,src_len,head_dim

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) ## bsz*num_head,tgt_len,src_len
        attn_weights += bias.view(bsz*self.num_heads,tgt_len,src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class KnowledgeAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        
        self.knowledge_attention_type = config.knowledge_attention_type
        if self.knowledge_attention_type == 'single_head':
            self.beta = nn.Parameter(torch.tensor(1.),requires_grad=True)
            self.key_proj = nn.Linear(config.hidden_size,config.hidden_size)
        elif self.knowledge_attention_type == 'multi_head':
            self.beta = nn.Parameter(torch.ones(config.num_attention_heads),requires_grad=True)
            self.biased_multihead_attn = BiasedMultiHeadAttention(config.hidden_size,config.num_attention_heads,config.attention_probs_dropout_prob)
        elif self.knowledge_attention_type == 'sentence_level':
            pass
        

    ## single head
    def forward(
        self,
        hidden_states, # [bs,seq_len,hidden_size]
        attention_mask, # [bs,seq_len]
        pooled_hidden_states, # [bs,query_size]
        kg_key, # [bs,k,query_size]
        kg_value, # [bs,k,hidden_size]
    ):
        if self.knowledge_attention_type == 'single_head':
            ## first compute the relevance between pooled_hidden_states and kg_key
            bs,seq_len,d_model = hidden_states.shape
            sentence_sim = F.cosine_similarity(pooled_hidden_states.unsqueeze(1),kg_key,dim=-1) ## [bs,k]
            attention_probs = torch.bmm(hidden_states,self.key_proj(kg_value).permute(0,2,1)) ## [bs,seq_len,k]
            attention_probs = self.beta * sentence_sim[:,None,:].expand(-1,seq_len,-1) # [bs,seq_len,k]
            attention_probs = F.softmax(attention_probs,dim=-1) # [bs,seq_len,k]
            hidden_states = torch.bmm(hidden_states,attention_probs)
            return hidden_states # [bs,seq_len,d_model]
        elif self.knowledge_attention_type == 'multi_head':
            bs,seq_len,d_model = hidden_states.shape
            num_attn_heads = self.beta.shape[0]
            sentence_sim = F.cosine_similarity(pooled_hidden_states.unsqueeze(1),kg_key,dim=-1) ## [bs,k]
            _, k = sentence_sim.shape
            bias = self.beta[None,:,None,None].expand(bs,-1,seq_len,k) * sentence_sim[:,None,None,:].expand(-1,num_attn_heads,seq_len,-1) # bs,num_heads,seq_len,k
            hidden_states = self.biased_multihead_attn(hidden_states=hidden_states,key_value_states=kg_value,bias=bias)
            return hidden_states # [bs,seq_len,d_model]
        elif self.knowledge_attention_type == 'sentence_level':
            bs,seq_len,d_model = hidden_states.shape
            sentence_sim = F.cosine_similarity(pooled_hidden_states.unsqueeze(1),kg_key,dim=-1) ## [bs,k]
            sentence_sim = F.softmax(sentence_sim,dim=-1)
            hidden_states = torch.bmm(sentence_sim.unsqueeze(1),kg_value).expand(-1,seq_len,-1)
            return hidden_states
        

class MemoryLayer(nn.Module):
    def __init__(self,config,retriever) -> None:
        super().__init__()
        self.config = config
        self.retriever = retriever
        self.pooler = Pooler(config.memory_pooler_type,config.hidden_size)
        self.know_attn = KnowledgeAttention(config)
        self.query_proj = nn.Linear(config.hidden_size,config.query_size)
        if config.query_batchnorm:
            self.batch_norm = nn.BatchNorm1d(num_features=config.query_size)
            
    
    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        pooled_hidden_states = self.pooler(hidden_states,attention_mask)
        pooled_hidden_states = self.query_proj(pooled_hidden_states)
        if hasattr(self,"batch_norm"):
            pooled_hidden_states = self.batch_norm(pooled_hidden_states)
        retriever_output = self.retriever(pooled_hidden_states)
        kg_key,kg_value,I = retriever_output.key,retriever_output.value,retriever_output.I
        hidden_states = self.know_attn(hidden_states,attention_mask,pooled_hidden_states,kg_key,kg_value)
        return MemoryLayerOutput(
            hidden_states=hidden_states,
            I=I,
        )
        


class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        assert act in ACT2FN, "Activation function is not found in activation dictionary."
        self.act_fn = ACT2FN[act]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #if not self.bias is None:
        #    return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        #else:
        return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class RetrievalBertLayer(nn.Module):
    
    def __init__(self,config,retriever) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        if config.memory_layer_ffn_dim > 0:
            self.ffn = nn.Sequential(
                LinearActivation(config.hidden_size,config.memory_layer_ffn_dim,config.hidden_act),
                nn.Linear(config.memory_layer_ffn_dim,config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )
        self.config = config
        self.memory_layer = MemoryLayer(config,retriever)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=None,
        *args,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )

        ## Self-Attention
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
                
        ## External Memory:TODO
        # mem_output,retrieved_index = self.memory_layer(_hidden_states,attention_mask)
        mem_output = self.memory_layer(attention_output,attention_mask)

        
        ## FFN
        if self.config.memory_layer_ffn_dim > 0:
            ffn_output = self.ffn(attention_output)    
            hidden_states = self.LayerNorm(attention_output + ffn_output + mem_output.hidden_states)
        else:
            hidden_states = self.LayerNorm(attention_output + mem_output.hidden_states)
        
        outputs = (hidden_states,)+outputs
        
        # return outputs
        return (hidden_states,mem_output.I)

class DocumentEncoder(nn.Module):
    def __init__(self,config,embedding=None,bert_layer=None) -> None:
        super().__init__()
        ## bert_layer is for multitaks learning
        self.document_encoder_type = config.document_encoder_type
        if config.document_encoder_type == 'word_embedding':
            self.embedding = embedding
            self.pooler = Pooler(config.document_encoder_pooler_type,config.hidden_size)
        elif config.document_encoder_type.startswith("first_"):
            n = int(config.document_encoder_type.split("_")[-1])
            self.embedding = embedding
            self.bert_encoder = nn.ModuleList(bert_layer[:n])
            self.pooler = Pooler(config.document_encoder_pooler_type,config.hidden_size)
    
    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        if self.document_encoder_type == 'word_embedding':
            hidden_states = self.embedding(input_ids)
            hidden_states = self.pooler(hidden_states,attention_mask)
        elif self.document_encoder_type.startswith('first_'):
            hidden_states = self.embedding(input_ids)
            hidden_states = self.bert_encoder(input_ids,attention_mask)
            hidden_states = self.pooler(hidden_states)
        else:
            raise NotImplementedError
        return hidden_states

class RetrievalBertEncoder(nn.Module):
    def __init__(self,config,embedding=None) -> None:
        super().__init__()
        self.config = config
        bert_layers = []
        memory_layers = []
        for _ in range(config.num_hidden_layers-config.num_memory_layers):
            bert_layers.append(BertLayer(config))
        self.retriever = Retriever(config,embedding,bert_layers)
        for _ in range(config.num_memory_layers):
            memory_layers.append(RetrievalBertLayer(config,self.retriever))
        self.layer = nn.ModuleList([*bert_layers,*memory_layers])
        
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_I = []

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if isinstance(layer_module,RetrievalBertLayer):
                all_I.append(layer_outputs[1])
        #     if use_cache:
        #         next_decoder_cache += (layer_outputs[-1],)
        #     if output_attentions:
        #         all_self_attentions = all_self_attentions + (layer_outputs[1],)
        #         if self.config.add_cross_attention:
        #             all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)
        if self.config.return_I:
            all_I = torch.stack(all_I).permute(1,0,2) ## [batch_size,num_layer,k]
        else:
            all_I = None
        return RetrievalBertLayerOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            I = all_I,
        )
  
class RetrievalBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = RetrievalBertEncoder(config,self.embeddings)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_document_encoder(self):
        return self.encoder.retriever

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # print("in the retrievalbert")
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )
        #print("pooled_output",pooled_output)
        return RetrievalBertModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            I = encoder_outputs.I,
        )

class RetrievalMixIn:
    def get_document_encoder(self):
        return self.bert.encoder.retriever
    def get_tokenized_document(self):
        return self.bert.encoder.retriever.tokenized_documents
    def update_memory(self,k,v=None,value_device='cpu'):
        self.bert.encoder.retriever.keys = torch.tensor(k,dtype=torch.float16).to(self.bert.embeddings.word_embeddings.weight.device)
        if v is not None:
            self.bert.encoder.retriever.values = torch.tensor(v,dtype=torch.float16).to(value_device)
            if value_device != 'cpu' and self.bert.encoder.retriever.keys.device != self.bert.encoder.retriever.values.device:
                self.bert.encoder.retriever.keys = self.bert.encoder.retriever.keys.to(self.bert.encoder.retriever.values.device)

## changed ignore index to -1 to be compatible with lddl dataloader
## change from next_sentence_label to next_sentence_labels
class BertForPreTrainingModified(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RetrievalBertForPreTraining(BertForPreTrainingModified,RetrievalMixIn):
    def __init__(self, config):
        super().__init__(config)    
        self.bert = RetrievalBertModel(config,add_pooling_layer=False)

class RetrievalBertForMaskedLM(BertForMaskedLM,RetrievalMixIn):
    def __init__(self,config):
        super().__init__(config)
        self.bert = RetrievalBertModel(config,add_pooling_layer=False)
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # print("masked_lm_loss",masked_lm_loss)
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RetrievalBertForSequenceClassification(BertForSequenceClassification,RetrievalMixIn):
    def __init__(self,config):
        super().__init__(config)
        self.bert = RetrievalBertModel(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RetrievalSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            I = outputs.I, #[num_memory_layers,bs,k]
        )

class RetrievalBertForQuestionAnswering(BertForQuestionAnswering,RetrievalMixIn):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RetrievalBertModel(config)


class RetrievalBertForTokenClassification(BertForTokenClassification,RetrievalMixIn):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RetrievalBertModel(config)

class RetrievalBertForYesno(RetrievalMixIn,BertPreTrainedModel):
    def __init__(self,config) -> None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1) # self.classifier
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.bert = RetrievalBertModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]        # use [CLS] pooled output

        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        # logits = self.sigmoid(logits)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.to(torch.float)
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


