# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

from lama.batch_eval_KB_completion import main as run_evaluation
from lama.batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

LMs = [
    # {
    #     "lm": "transformerxl",
    #     "label": "transformerxl",
    #     "models_names": ["transformerxl"],
    #     "transformerxl_model_name": "transfo-xl-wt103",
    #     "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
    #     "elmo_vocab_name": "vocab-2016-09-10.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original",
    #     "elmo_warm_up_cycles": 10,
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo5B",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
    #     "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
    #     "elmo_warm_up_cycles": 10,
    # },
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "../lama_pretrained_models/bert/cased_L-12_H-768_A-12",
    },
    # {
    #     "lm": "bert",
    #     "label": "bert_large",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-large-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    # },
]

Uncased_LMs = [
    {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-uncased",
        "bert_model_dir": "../lama_pretrained_models/bert/uncased_L-24_H-1024_A-16",
    },
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-uncased",
        "bert_model_dir": "../lama_pretrained_models/bert/uncased_L-12_H-768_A-12",
    },
    {
        "lm": "gpt",
        "label": "gpt",
        "models_names": ["gpt"],
        "gpt_model_name": "openai-gpt",
        "gpt_model_dir": "../lama_pretrained_models/gpt/openai-gpt/",
    },
]

Retrieval_LMs = [
    {
        "lm":"retrieval_bert",
        "label":"retrieval_bert",
        "models_names":["retrieval_bert"],
        # "model_name":"retrieval_bert",
        "model_dir":"TODO"
    }
]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    all_Precision1_samples = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    # results_file = open("output/last_results.csv", "w+")
    is_uncased = False
    # is_uncased = True
    for relation in relations:
        # pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            # "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt" if not is_uncased else "pre-trained_language_models/common_vocab_lowercased.txt",
            "common_vocab_filename":"../lama_pretrained_models/common_vocab_cased.txt" if not is_uncased else "../lama_pretrained_models/common_vocab_lowercased.txt", 
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": True if is_uncased else False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        # print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue
        
        # "models_names": ["bert"]
        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1,real_samples,supposed_samples = run_evaluation(args, shuffle_data=False, model=model)
        if "TREx" not in data_path_pre:
            print("****** Results *******")
            print(args.dataset_filename)
            print(model_type_name)
            print(args.common_vocab_filename)
            print("ori dataset size",supposed_samples)
            print("real dataset size",real_samples)
            print("P@1 : {}".format(Precision1), flush=True)

        all_Precision1.append(Precision1)
        all_Precision1_samples.append(real_samples)

        # results_file.write(
        #     "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        # )
        # results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            # data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(real_samples)

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    weighted_mean_p1 = sum([p*n for p,n in zip(all_Precision1,all_Precision1_samples)])/sum(all_Precision1_samples)
    print("@@@ {} - mean weighted P@1: {}".format(input_param["label"], weighted_mean_p1))
    # results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters(data_path_pre = "data/"):
    relations = [
        {
            "relation": "place_of_birth",
            "template": "The birth place of [X] is [Y] .",
            # "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },

        {
            "relation": "date_of_birth",
            "template": "The year when [X] is born in is [Y] .",
            # "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template":"[X] died in the place of [Y] .",
            # "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre += "Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def run_retrieval_LMs(parameters):
    for ip in Retrieval_LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

def run_bert_base(parameters):
    run_experiments(*parameters, input_param=LMs[3], use_negated_probes=False)

def run_all_Uncased_LMs(parameters):
    for ip in Uncased_LMs:
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cased",action='store_true')
    parser.add_argument("--uncased",action='store_true')
    parser.add_argument("--ckpt")
    parser.add_argument("--data_dir",default="../lama_data/")
    args = parser.parse_args()

    if args.cased:
        exp = run_all_LMs
    elif args.uncased:
        exp = run_all_Uncased_LMs
    elif args.ckpt is not None:
        exp = run_retrieval_LMs
        Retrieval_LMs[0]['model_dir'] = args.ckpt    


    print("1. Google-RE")
    parameters = get_GoogleRE_parameters(args.data_dir)
    exp(parameters)
    
    print("2. T-REx")
    parameters = get_TREx_parameters(args.data_dir)
    exp(parameters)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters(args.data_dir)
    exp(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters(args.data_dir)
    exp(parameters)