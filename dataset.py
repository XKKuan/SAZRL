import random
import json
import numpy as np
from utils import *

class Data():
    def __init__(self,data_path='Data',data_name = 'Wiki-ZS',candidate = True,inverse = False,):
        
        self.data_name = data_name
        self.path = data_path + '/' + data_name + '/'

        
        self.train_ent, self.train_rel, self.train_triple = self.read_triple(self.path,'train',inverse = inverse)
        self.valid_ent, self.valid_rel, self.valid_triple = self.read_triple(self.path,'valid',inverse = inverse)
        self.test_ent, self.test_rel, self.test_triple = self.read_triple(self.path,'test',inverse = inverse)
        
        self.entities = remove_duplicate(self.train_ent + self.valid_ent +self.test_ent)
        self.relations = remove_duplicate(self.train_rel + self.valid_rel +self.test_rel)
        
        self.ent2id = {ent: idx for idx, ent in enumerate(self.entities)}
        self.rel2id = {rel: idx for idx, rel in enumerate(self.relations)}
        
        self.train2id = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in self.train_triple]
        self.valid2id = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in self.valid_triple]
        self.test2id = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in self.test_triple]
        
        self.valid_filter = self.filter(self.valid2id)
        self.test_filter = self.filter(self.test2id)
        
        self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id)
        
        if candidate:
            self.valid_candidate_id, self.test_candidate_id = self.get_candidate(self.path,self.ent2id,self.rel2id,self.data_name)
        
        
    def read_triple(self, path, data_type, inverse = False):
        
        ent, rel, triple = [], [], []
        with open(path + data_type + '.txt', 'r') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                ent.append(h)
                ent.append(t)
                rel.append(r)
                triple.append([h,r,t])
                if inverse :
                    rel.append(r+'_inv')
                    triple.append([t,r+'_inv',h])
                
        ent = remove_duplicate(ent)
        rel = remove_duplicate(rel)
        
        print(f"-----{data_type.capitalize()} Data Statistics-----")
        print(f"{len(ent)} entities, {len(rel)} relations, {len(triple)} triplets")
        
        return ent, rel, triple
    
    def filter(self, triples):
        
        filter_dict = {}
        
        for triple in triples:
            h,r,t = triple
            if (h,r,'_') not in filter_dict:
                filter_dict[(h,r,'_')] = [t]
            else:
                filter_dict[(h,r,'_')].append(t)
            
        return filter_dict
    
    def get_candidate(self,path,ent2id,rel2id,data_name):
        
        if data_name == 'Wiki-ZS' or data_name == 'FB-RZS':
            valid_candidate = json.load(open(path+"valid_hr_candidates.json"))
            test_candidate = json.load(open(path+"test_hr_candidates.json"))

        if data_name == 'NELL-ZS':
            valid_candidate = json.load(open(path+"valid_hrt_candidates.json"))
            test_candidate = json.load(open(path+"test_hrt_candidates.json"))

        
        valid_candidate_id = self.candidate2id(valid_candidate,ent2id,rel2id)
        test_candidate_id = self.candidate2id(test_candidate,ent2id,rel2id)
            
        return valid_candidate_id, test_candidate_id
        
    def candidate2id(self,candidate,ent2id,rel2id):
         
        candidate_id = {}
        
        for key, values in candidate.items():
            
            key_list = key.split("\t")
            key_id = tuple([ent2id[key_list[i]] if i !=1 else rel2id[key_list[i]] for i in range(len(key_list))])

            value_id = [ent2id[v] for v in values]
            candidate_id[key_id] = value_id
        
        return candidate_id
