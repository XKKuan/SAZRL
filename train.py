import os
import torch
import random
import numpy as np
from loss import *
from Parser import parse
from tqdm import tqdm
from model import SAZRL
from dataset import Data
from utils import *
from generate_G import *
from evaluation import *
from torch.optim.lr_scheduler import CosineAnnealingLR

OMP_NUM_THREADS = 8
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

args = parse()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + '/' + args.data_name + '/'

if args.write:
    os.makedirs(f"./ckpt/{args.data_name}", exist_ok=True)

total_data = Data(args.data_path, args.data_name, args.candidate)
train_data = total_data.train2id

device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")

model = SAZRL(total_data.num_ent, args.dim_ent, total_data.num_rel, args.dim_rel,
              args.hid_dim_ratio_rel, args.num_layer_rel, args.B,
              device=device, num_head=8, bias=True, auto=args.AutoRelation).to(device)

if args.loss == 'margin':
    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin, reduction='mean')
elif args.loss == 'adv':
    loss_fn = AdvLoss(margin=args.margin, temperature=1.0)
elif args.loss == 'crr':
    loss_fn = CRRLoss(t=3.0, p=0.1)
elif args.loss == 'soft':
    loss_fn = SoftplusLoss(adv_temperature=args.adv_temperature)
elif args.loss == 'sig':
    loss_fn = SigmoidLoss(adv_temperature=args.adv_temperature)

optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
if args.schedul:
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-3)

pbar = tqdm(range(args.num_epoch))
best = {'MRR': 0, 'Hits10': 0, 'Hits5': 0, 'Hits1': 0}

for epoch in pbar:
    model.train()
    losses = []

    gen = batch_generator2(train_data, p_num=args.n_batch)
    for pos in gen:
        optimizer.zero_grad()

        neg = generate_neg(pos, total_data, num_neg=args.num_neg)

        if args.AutoRelation:
            initial_ent = torch.cat([pos[:, 0], pos[:, 2]]).int().tolist()
            support_rel = set(pos[:, 1].int().tolist())
            support_ent = sample_entity(train_data, initial_ent, thr=50, n_hop=args.n_hop)
            support_triple = [(h, r, t) for h, r, t in train_data if h in support_ent and r not in support_rel and t in support_ent]

            query_triple_h = pos[:, 1:].numpy()
            query_triple_t = pos[:, :2].numpy()
            relation_graph = generate_relation_graph(np.array(support_triple), query_triple_h, query_triple_t, total_data)
            relation_graph_discrete = get_relation_triplets(relation_graph, args.B).to(device)

            emb_ent, emb_rel = model(relation_graph_discrete)
        else:
            emb_ent, emb_rel = model(None)

        pos = pos.to(device)
        neg = neg.to(device)

        pos_scores = model.get_score(emb_ent, emb_rel, pos)
        neg_scores = model.get_score(emb_ent, emb_rel, neg)

        if args.loss == 'margin':
            loss = loss_fn(pos_scores.unsqueeze(1).repeat(1, args.num_neg).flatten(), neg_scores, torch.ones_like(neg_scores))
        else:
            loss = loss_fn(pos_scores.unsqueeze(1).repeat(1, args.num_neg).flatten(), neg_scores, args.num_neg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=False)
        optimizer.step()
        losses.append(loss.item())

    pbar.set_description(f"loss {np.mean(losses)}")

    if args.schedul:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")
        scheduler.step()

    if ((epoch + 1) % args.valid_epochs) == 0:
        print(f"--------Epoch:{epoch + 1}--------")

        MR, MRR, Hits10, Hits5, Hits1 = evaluate(model=model, total_data=total_data, task='test', support_graph=train_data, args=args)
        MRR, Hits10, Hits5, Hits1 = round(MRR, 3), round(Hits10, 3), round(Hits5, 3), round(Hits1, 3)
        print("--------Test LP--------")
        current = {'MRR': MRR, 'Hits10': Hits10, 'Hits5': Hits5, 'Hits1': Hits1}
        print('Current:', current)
