import numpy as np
import json
from transformers import BertTokenizerFast
toker = BertTokenizerFast.from_pretrained("/data/pretrained_model/bert_base_uncased")
def get_jsonl(f):
    import json
    return [json.loads(x) for x in open(f).readlines()]
dataset = 'wnli'
data = get_jsonl("../data/"+dataset+"/train.jsonl")
print(len(data))
print(data[0].keys())
print(data[0])
to_be_encoded = []
for x in data:
    temp = []
    # temp.append(x['text'])
    # temp.append(x['question1'])
    temp.append(x['sentence1'])
    additional_key = 'sentence2'
    if x[additional_key]:
        temp[0]+=x[additional_key]
    temp.append(str(x['label']))
    to_be_encoded.append(temp)
print(to_be_encoded[0])
tokenized_data = toker(to_be_encoded,max_length=288,truncation='only_first',padding='max_length').input_ids
np.save(open("../tokenized_data/tokenized_"+dataset+".npy",'wb'),np.array(tokenized_data,dtype=np.int16))