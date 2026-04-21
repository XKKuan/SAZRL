import torch
from utils import *
from generate_G import *
import numpy as np
import random
from tqdm import tqdm

def evaluate(model = None,total_data = None, task = 'valid',support_graph = [],args = None):
    
    with torch.no_grad():
        device = torch.device("cuda:"+str(args.cuda) if torch.cuda.is_available() else "cpu")
        model.eval()

        if task == 'test':
            query = total_data.test2id
        elif task == 'valid':
            query = total_data.valid2id

        ranks = []
        gen = batch_generator1(query, p_num=args.batch_test)
        for eval in gen:
            
            if args.AutoRelation:
                support_rel = set(eval[:, 1].int().tolist())#获取当前需要预测的关系
                
                # initial_ent = torch.cat([eval[:, 0], eval[:, 2]]).int().tolist()
                initial_ent = eval[:, 0].int().tolist()
                
                if args.subgraph:
                    support_ent = sample_entity(support_graph, initial_ent, thr = 50, n_hop=args.n_hop)
                    support_triple = [(h, r, t) for h, r, t in support_graph if h in support_ent and r not in support_rel and t in support_ent]
                else:
                    support_ent = initial_ent
                    support_triple = [(h, r, t) for h, r, t in support_graph if (h in support_ent or t in support_ent) and r not in support_rel]

                # query_triple_h = eval[:,1:].numpy()
                query_triple_h = None
                query_triple_t = eval[:,:2].numpy()
                
                relation_graph = generate_relation_graph(np.array(support_triple),query_triple_h, query_triple_t, total_data,mode='eval')
                relation_graph_discrete = get_relation_triplets(relation_graph,args.B).to(device)
                
                emb_ent, emb_rel = model(relation_graph_discrete)
            else:
                emb_ent, emb_rel = model(None)

            for triplet in eval:

                triplet = triplet.unsqueeze(dim = 0)


                if task == 'valid':

                    filters = total_data.valid_filter[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]

                    if total_data.data_name == 'NELL-ZS':
                        candidates = total_data.valid_candidate_id[(int(triplet[0,0].item()),int(triplet[0,1].item()),int(triplet[0,2].item()))]
                    elif total_data.data_name == 'Wiki-ZS' or total_data.data_name == 'FB-RZS':
                        candidates = total_data.valid_candidate_id[(int(triplet[0,0].item()),int(triplet[0,1].item()))]
                
                elif task == 'test':

                    filters = total_data.test_filter[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]

                    if total_data.data_name == 'NELL-ZS':
                        candidates = total_data.test_candidate_id[(int(triplet[0,0].item()),int(triplet[0,1].item()),int(triplet[0,2].item()))]
                    elif total_data.data_name == 'Wiki-ZS' or total_data.data_name == 'FB-RZS':
                        candidates = total_data.test_candidate_id[(int(triplet[0,0].item()),int(triplet[0,1].item()))]
                
                candidates = list(set(candidates)|set(filters))
                
                corrupt = triplet.repeat(len(candidates), 1)
                
                corrupt[:,2] = torch.tensor(candidates)
                
                corrupt = corrupt.to(device)
                scores = model.get_score(emb_ent, emb_rel, corrupt)

                rank = get_rank(triplet,scores,filters,candidates,target = 2)

                ranks.append(rank)
        
        MR,MRR,Hits10,Hits5,Hits1 = get_metrics(ranks)
        return MR,MRR,Hits10,Hits5,Hits1    
