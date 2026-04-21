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

model.load_state_dict(torch.load('ckpt/Wiki-ZS/Best3.pkl', weights_only=True))

MR, MRR, Hits10, Hits5, Hits1 = evaluate(model=model, total_data=total_data, task='test', support_graph=train_data, args=args)

print("--------Test LP--------")
print(f"MR: {MR:.1f}")
print(f"MRR: {MRR:.3f}")
print(f"Hits@10: {Hits10:.3f}")
print(f"Hits@5: {Hits5:.3f}")
print(f"Hits@1: {Hits1:.3f}")
