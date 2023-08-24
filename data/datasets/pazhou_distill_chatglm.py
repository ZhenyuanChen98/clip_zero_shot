import os
from os.path import join
from re import L
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
from clip import tokenize
import re
from pycocotools.coco import COCO
from .data_helpers import *


def is_valid_sentence(sentence):
    words = sentence.split()
    for word in words:
        if len(word) > 77 or words.count(word) > 5:
            return False

    return True

@DATASET_REGISTRY.register()
class pazhou_distill_chatglm(DatasetBase):
    def __init__(self, cfg):
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = os.path.join(root, "A榜数据集/")
        with open(join(root, 'classes.txt'), 'r') as f:
            object_categories = f.readlines()
        object_categories = [i.strip() for i in object_categories]
        cls_num = len(object_categories)

        self.dataset_dir_coco = 'coco_stuff10k'
        root_coco = os.path.abspath(os.path.expanduser('/data/datasets/'))
        self.dataset_dir_coco = os.path.join(root_coco, self.dataset_dir_coco)

        coco_instance_json_file = os.path.join(self.dataset_dir_coco, "instances_val2014.json")

        coco = COCO(coco_instance_json_file)
        self.valset_ids = coco.getImgIds()
        

        instance_info = {}
        with open(coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']

        val_imgdir = [self.dataset_dir_coco + '/val2014/{}'.format(coco.loadImgs(ids = imgid)[0]['file_name']) for imgid in self.valset_ids]
        val_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
        for idx, imgid in enumerate(self.valset_ids):
            annIds = coco.getAnnIds(imgIds = imgid)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                tmp_idx = clsid2clsidx[ann['category_id']]
                val_label[idx, tmp_idx] = 1

        val = []
        for i in range(len(self.valset_ids)):
            item_ = Datum(impath=val_imgdir[i], label=val_label[i], classname='')
            val.append(item_)

        self.dataset_dir = os.path.join(root, 'dataset_A')

        with open(join(root, 'imnames_A.json'), 'r') as f:
            imnames_a = json.load(f)

        test = []
        for idx, imgid in enumerate(imnames_a):
            tmp_label = torch.zeros(cls_num)
            item_ = Datum(impath=join(root, 'dataset_A', imgid.split('/')[-1]), label=tmp_label, classname='')
            test.append(item_)

        # ===================  training captions
        with open(f'/data/clip_zero_shot/data/data_{cfg.data_size}.json', 'r') as f:
            texts_dict = [json.loads(line) for line in f]
        
        all_prompts = []
        all_labels = []

        for item in texts_dict:
            text = item["text"]
            labels = item["labels"]
            if not is_valid_sentence(text):
                pass # Skip tokenization
            else:
                tokenized_text = clip.tokenize(text)
                all_prompts.append(tokenized_text)

                cls_labels = torch.tensor([[0] * cls_num])
                
                for idx in labels:
                    cls_labels[0, idx] = 1
                all_labels.append(cls_labels)
            
        all_prompts = torch.cat(all_prompts)
        all_labels = torch.cat(all_labels)
        
        print('===== chatglm generate {} sentences ====='.format(all_prompts.shape[0]), all_prompts.shape, all_labels.shape)
        
        # #
        train = []
        if not cfg.TRAIN.IF_ablation:
            for i in range(all_prompts.shape[0]):
                item_ = (all_prompts[i], all_labels[i])
                train.append(item_)
        print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(train)))


        super().__init__(train_x=train, val=val, test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
