import os
import sys
sys.path.insert(0, '/data/clip_zero_shot/')
from codes.baseline import load_classes
import random
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from constant import DATA_DIR

#import nltk
#from nltk.corpus import words
#nltk.download('words')

import os
from typing import Dict, Tuple, Union, Optional

from torch.nn import Module
from transformers import AutoModel
from googletrans import Translator
import inflection
import re

coco_classname_synonyms = [
    ['person', 'human', 'people', 'man', 'woman', 'passenger', 'girl', 'boy'], 
    ['bicycle', 'bike', 'cycle'],
    ['car', 'taxi', 'auto', 'automobile', 'motor car'], 
    ['motor bike', 'motor cycle', 'motorbike'], 
    ['aeroplane', "air craft", "jet", "plane", "air plane"], 
    ['bus', 'autobus', 'coach', 'charabanc', 'double decker', 'jitney', 'motor bus', 'motor coach', 'omnibus'],
    ['train', 'rail way', 'railroad'], 
    ['truck'],
    ['boat', 'raft', 'dinghy'],
    ['traffic light'],
    ['fire hydrant', 'fire tap', 'hydrant'],
    ['stop sign', 'halt sign'],
    ['parking meter'],
    ['bench'],
    ['bird'],
    ['cat', 'kitty'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['horse', 'colt', 'equus'],
    ['sheep'],
    ['cow'],
    ['elephant'],
    ['bear'],
    ['zebra'],
    ['giraffe', 'camelopard'],
    ['backpack', 'back pack', 'knapsack', 'packsack', 'rucksack', 'haversack'],
    ['umbrella'],
    ['handbag', 'hand bag', 'pocketbook', 'purse'],
    ['tie', 'necktie'],
    ['suitcase'],
    ['frisbee'],
    ['skis', 'ski'],
    ['snowboard'],
    ['sports ball', 'sport ball', 'ball', 'football', 'soccer', 'tennis', 'basketball', 'baseball'],
    ['kite'],
    ['baseball bat', 'baseball game bat'],
    ['baseball glove', 'baseball mitt', 'baseball game glove'],
    ['skateboard'],
    ['surfboard'],
    ['tennis racket'],
    ['bottle'],
    ['wine glass', 'vino glass'],
    ['cup'],
    ['fork'],
    ['knife'],
    ['spoon'],
    ['bowl'],
    ['banana'],
    ['apple'],
    ['sandwich'],
    ['orange'],
    ['broccoli'],
    ['carrot'],
    ['hot dog'],
    ['pizza'],
    ['donut', 'doughnut'],
    ['cake'],
    ['chair', 'arm chair'],
    ['couch', 'sofa'],
    ['potted plant', 'house plant', 'bonsai', 'pot plant', 'pottedplant'],
    ['bed'],
    ['dining table', 'dinner table', 'table', 'din table', 'diningtable'], 
    ['toilet', 'commode'],
    ['tv', 'tvmonitor', 'monitor', 'television', 'telly'],
    ['laptop'],
    ['mouse'],
    ['remote'],
    ['keyboard'],
    ['cell phone', 'phone', 'mobile phone'],
    ['microwave'],
    ['oven', 'roaster'],
    ['toaster'],
    ['sink'],
    ['refrigerator', 'icebox'],
    ['book'],
    ['clock'],
    ['vase'],
    ['scissors'],
    ['teddy bear', 'teddy'],
    ['hair drier', 'blowing machine', 'hair dryer', 'dryer', 'blow dryer', 'blown dry', 'blow dry'],
    ['toothbrush'],
]

def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model

def init_chatglm():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0")
    #model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
    model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=2)
    model = model.eval()
    return model, tokenizer


# def generate_by_chatglm(model, tokenizer, categories):
#     ## noq

#     datas = []

#     for _ in range(10000):
#         sample_labels = [random.randint(0, len(categories)-1) for _ in range(random.randint(1, 6)) ]
#         sample_categories = [categories[label]  for label in sample_labels]
#         history = []

#         for _ in range(100):
#             request = "Please use {} as the topic, imagine and briefly describe a photo, which is different from what you imagined and described before.".format(",".join(sample_categories))
#             response, history = model.chat(tokenizer, request, history=history)
#             datas.append(dict(text=response, labels=sample_labels) )

#     open(DATA_DIR / "data.json", "w").writelines([json.dumps(data)+"\n" for data in datas])

# def generate_by_chatglm(model, tokenizer, categories):
#     with open(DATA_DIR / "data.json", "w") as f:
#         for _ in range(10000):
#             # sample_labels = [random.randint(0, len(categories)-1) for _ in range(random.randint(1, 5))]
#             # sample_categories = [categories[label] for label in sample_labels]
#             # other_labels = [i for i in range(len(categories)) if i not in sample_labels]
#             # other_categories = [categories[label] for label in other_labels]

#             num_labels = random.randint(1, 5)
#             sample_labels = random.sample(range(len(categories)), num_labels)
#             sample_categories = [categories[label] for label in sample_labels]
#             other_labels = [i for i in range(len(categories)) if i not in sample_labels]
#             other_categories = [categories[label] for label in other_labels]
            
#             #history = []
#             for _ in range(10):
#                 # request = "Please use {} nouns as the topic, and make sure to include {} nouns. Do not include {} nouns or their synonyms. \
#                 #     Generate 10 sentences in different scenes, and ensure they do not appear in Chinese.".format(", ".join(sample_categories), ", ".join(sample_categories), ", ".join(other_categories))
#                 request = "Please create 10 sentences in different scenarioson the following instructions: \
#                     Use the nouns: {} as the main topic. Ensure that all of these nouns are included: {}. Do not include any of these nouns or their synonyms: {}. \
#                     The generated sentences must not be in Chinese.".format(", ".join(sample_categories), ", ".join(sample_categories), ", ".join(other_categories))

#                 response, history = model.chat(tokenizer, request, history=[])
#                 print(response)
#                 print(sample_categories)
#                 # Split response into lines
#                 response_lines = response.split('\n')
#                 # Save each line as a separate data
#                 for line in response_lines:
#                     cleaned_line = line[line.find('.')+2:]
#                     data = dict(text=cleaned_line, labels=sample_labels)
#                     f.write(json.dumps(data) + "\n")

# def generate_by_chatglm(model, tokenizer, categories):
#     translator = Translator()
    
#     with open(DATA_DIR / "data.json", "w") as f:
#         for _ in range(10000):
#             num_labels = random.randint(1, 5)
#             sample_labels = random.sample(range(len(categories)), num_labels)
#             sample_categories = [categories[label] for label in sample_labels]
#             other_labels = [i for i in range(len(categories)) if i not in sample_labels]
#             other_categories = [categories[label] for label in other_labels]
#             print(sample_categories)
#             for _ in range(10):
#                 request = "Please create 10 sentences in different scenarios on the following instructions: \
#                 Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Ensure that all of these nouns are included: {}. Do not include any of these nouns or their synonyms: {}. \
#                 ".format(", ".join(sample_categories), ", ".join(sample_categories), ", ".join(other_categories))
#                 response, history = model.chat(tokenizer, request, history=[])
#                 response_lines = response.split('\n')
#                 for line in response_lines:
#                     # Translate line to English if it is in Chinese
#                     translation = translator.translate(line)
#                     if translation.src == 'zh-CN':
#                         line = translation.text
#                     print(line)
#                     cleaned_line = line[line.find('.')+2:]
#                     data = dict(text=cleaned_line, labels=sample_labels)
#                     f.write(json.dumps(data) + "\n")

# def generate_by_chatglm(model, tokenizer, categories):
#     translator = Translator()
    
#     with open(DATA_DIR / "data.json", "w") as f:
#         for _ in range(10000):
#             num_labels = random.randint(1, 5)
#             sample_labels = random.sample(range(len(categories)), num_labels)
#             sample_categories = [categories[label] for label in sample_labels]
#             #other_labels = [i for i in range(len(categories)) if i not in sample_labels]
#             #other_categories = [categories[label] for label in other_labels]
#             print(sample_categories)
#             for _ in range(10):
#                 request = "Please create 10 sentences in different scenarios on the following instructions: \
#                 Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Please do not exceed 50 words. \
#                 ".format(", ".join(sample_categories))
#                 response, history = model.chat(tokenizer, request, history=[])
#                 response_lines = response.split('\n')
#                 for line in response_lines:
#                     # Translate line to English if it is in Chinese
#                     translation = translator.translate(line)
#                     if translation.src == 'zh-CN':
#                         line = translation.text
                    
#                     cleaned_line = line[line.find('.')+2:] 
#                     # Find the corresponding category index for words in cleaned_line
#                     sample_labels_tmp = []
#                     words = cleaned_line.split(' ')
#                     for index, category in enumerate(coco_classname_synonyms):
#                         for word in words:
#                             if word.lower() in category:
#                                 sample_labels_tmp.append(index)
                    
#                     # Keep only unique values in sample_labels_tmp
#                     sample_labels_tmp = list(set(sample_labels_tmp))
#                     if len(sample_labels_tmp) != 0: 
#                         print(cleaned_line, sample_labels_tmp)
#                         data = dict(text=cleaned_line, labels=sample_labels_tmp)
#                         f.write(json.dumps(data) + "\n")

# def generate_by_chatglm(model, tokenizer, categories):   
#     with open(DATA_DIR / "data.json", "w") as f:
#         for _ in range(10000):
#             num_labels = random.randint(1, 5)
#             sample_labels = random.sample(range(len(categories)), num_labels)
#             sample_categories = [categories[label] for label in sample_labels]
#             print(sample_categories)
#             for _ in range(10):
#                 request = "Please create 10 sentences in different scenarios on the following instructions: \
#                 Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Please do not exceed 50 words. \
#                 ".format(", ".join(sample_categories))
#                 response, history = model.chat(tokenizer, request, history=[])
#                 response_lines = response.split('\n')
#                 for line in response_lines:                    
#                     cleaned_line = line[line.find('.')+2:]

#                     # Check if cleaned_line is still in Chinese, if so, skip
#                     if any(u'\u4e00' <= c <= u'\u9fff' for c in cleaned_line):
#                         continue
                    
#                     # Find the corresponding category index for words in cleaned_line
#                     sample_labels_tmp = []
#                     words = cleaned_line.split(' ')
#                     for index, category in enumerate(coco_classname_synonyms):
#                         for word in words:
#                             if word.lower() in category:
#                                 sample_labels_tmp.append(index)
                    
#                     # Keep only unique values in sample_labels_tmp
#                     sample_labels_tmp = list(set(sample_labels_tmp))
#                     if len(sample_labels_tmp) != 0: 
#                         print(cleaned_line, sample_labels_tmp)
#                         data = dict(text=cleaned_line, labels=sample_labels_tmp)
#                         f.write(json.dumps(data) + "\n")

# def generate_by_chatglm(model, tokenizer, categories):   
#     with open(DATA_DIR / "data.json", "w") as f:
#         for _ in range(10000):
#             num_labels = random.randint(1, 5)
#             sample_labels = random.sample(range(len(categories)), num_labels)
#             sample_categories = [categories[label] for label in sample_labels]
#             print(sample_categories)
#             for _ in range(10):
#                 request = "Please create 10 sentences in different scenarios on the following instructions: \
#                 Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Please do not exceed 50 words. \
#                 ".format(", ".join(sample_categories))
#                 response, history = model.chat(tokenizer, request, history=[])
#                 response_lines = response.split('\n')
#                 for line in response_lines:                    
#                     cleaned_line = line[line.find('.')+2:]

#                     # Check if cleaned_line is still in Chinese, if so, skip
#                     if any(u'\u4e00' <= c <= u'\u9fff' for c in cleaned_line):
#                         continue
                    
#                     # Find the corresponding category index for words in cleaned_line
#                     sample_labels_tmp = []
#                     words = cleaned_line.split(' ')
#                     for index, category in enumerate(coco_classname_synonyms):
#                         for word in words:
#                             if inflection.singularize(word.lower()) in [inflection.singularize(w) for w in category]:
#                                 sample_labels_tmp.append(index)

#                     # Keep only unique values in sample_labels_tmp
#                     sample_labels_tmp = list(set(sample_labels_tmp))
#                     if len(sample_labels_tmp) != 0: 
#                         print(cleaned_line, sample_labels_tmp)
#                         data = dict(text=cleaned_line, labels=sample_labels_tmp)
#                         f.write(json.dumps(data) + "\n")



def generate_by_chatglm(model, tokenizer, categories):  
     
    with open(DATA_DIR / "data_50w.json", "a") as f:
        for _ in range(5000):
            num_labels = random.randint(1, 5)
            sample_labels = random.sample(range(len(categories)), num_labels)
            sample_categories = [categories[label] for label in sample_labels]
            print(sample_categories, sample_labels)
            for _ in range(10):
                request = "Please create 10 sentences in different scenarios on the following instructions: \
                Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Please do not exceed 50 words. \
                ".format(", ".join(sample_categories))
                response, history = model.chat(tokenizer, request, history=[])
                response_lines = response.split('\n')
                for line in response_lines:                    
                    cleaned_line = line[line.find('.')+2:]

                    # Check if cleaned_line is still in Chinese, if so, skip
                    if any(u'\u4e00' <= c <= u'\u9fff' for c in cleaned_line):
                        continue
                    
                    # Find the corresponding category index for words in cleaned_line
                    # 根据词组或单词是否包含空格，构建一个正则表达式模式去匹配 clean_line。如果成功匹配，将对应的索引添加到 sample_labels_tmp 列表。
                    sample_labels_tmp = []
                    for index, category in enumerate(coco_classname_synonyms):
                        for w in category:
                            singularized_w = inflection.singularize(w.lower())
                            pattern = r'\b{}\b'.format(singularized_w) if ' ' not in singularized_w else re.escape(singularized_w)
                            if re.search(pattern, cleaned_line.lower()):
                                sample_labels_tmp.append(index)

                    # Keep only unique values in sample_labels_tmp
                    sample_labels_tmp = list(set(sample_labels_tmp))
                    if len(sample_labels_tmp) != 0: 
                        print(cleaned_line, sample_labels_tmp)
                        data = dict(text=cleaned_line, labels=sample_labels_tmp)
                        f.write(json.dumps(data) + "\n")


# def generate_by_chatglm(model, tokenizer, categories):  
#     datas = [] 
#     for _ in range(10000):
#         num_labels = random.randint(1, 5)
#         sample_labels = random.sample(range(len(categories)), num_labels)
#         sample_categories = [categories[label] for label in sample_labels]
#         print(sample_categories, sample_labels)
#         for _ in range(10):
#             request = "Please create 10 sentences in different scenarios on the following instructions: \
#             Start each sentence with 'A photo of'. Use the nouns: {} as the main topic. Please do not exceed 50 words. \
#             ".format(", ".join(sample_categories))
#             response, history = model.chat(tokenizer, request, history=[])
#             response_lines = response.split('\n')
#             for line in response_lines:                    
#                 cleaned_line = line[line.find('.')+2:]

#                 # Check if cleaned_line is still in Chinese, if so, skip
#                 if any(u'\u4e00' <= c <= u'\u9fff' for c in cleaned_line):
#                     continue
                
#                 # Find the corresponding category index for words in cleaned_line
#                 # 根据词组或单词是否包含空格，构建一个正则表达式模式去匹配 clean_line。如果成功匹配，将对应的索引添加到 sample_labels_tmp 列表。
#                 sample_labels_tmp = []
#                 for index, category in enumerate(coco_classname_synonyms):
#                     for w in category:
#                         singularized_w = inflection.singularize(w.lower())
#                         pattern = r'\b{}\b'.format(singularized_w) if ' ' not in singularized_w else re.escape(singularized_w)
#                         if re.search(pattern, cleaned_line.lower()):
#                             sample_labels_tmp.append(index)

#                 # Keep only unique values in sample_labels_tmp
#                 sample_labels_tmp = list(set(sample_labels_tmp))
#                 if len(sample_labels_tmp) != 0: 
#                     print(cleaned_line, sample_labels_tmp)
#                     datas.append(dict(text=cleaned_line, labels=sample_labels_tmp))
                    
#     open(DATA_DIR / "data.json", "w").writelines([json.dumps(data)+"\n" for data in datas])


def generate_by_rule(categories):

    datas = []

    for idx in range(30000):
        sample_labels = [random.randint(0, len(categories)-1) for _ in range(random.randint(1, 6)) ]
        sample_categories = [categories[label]  for label in sample_labels]

        text = "there are {} in the photo".format(",".join(sample_categories))
        datas.append(
                dict(text=text, labels=sample_labels) )

    open(DATA_DIR / "data.json", "w").writelines([json.dumps(data)+"\n" for data in datas])

if __name__=="__main__":

    categories = load_classes(DATA_DIR / "classes" / "ZSMLL_classes.txt")
    model, tokenizer = init_chatglm()
    generate_by_chatglm(model,tokenizer,categories)









            
















