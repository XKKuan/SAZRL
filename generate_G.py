import math
import igraph
from utils import *
import numpy as np
from scipy.sparse import csr_matrix


def generate_relation_graph(support_triple, query_triple_h, query_triple_t, data, mode='train'):
    num_ent = data.num_ent
    num_rel = data.num_rel

    if mode == 'train':
        ind_h = np.concatenate((support_triple[:, :2], query_triple_t), axis=0)
        ind_t = np.concatenate((support_triple[:, 1:], query_triple_h), axis=0)
    elif mode == 'eval':
        ind_h = np.concatenate((support_triple[:, :2], query_triple_t), axis=0)
        ind_t = support_triple[:, 1:]

    E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, num_rel))
    E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, num_rel))

    diag_vals_h = E_h.sum(axis=1).A1
    diag_vals_h[diag_vals_h != 0] = 1 / (diag_vals_h[diag_vals_h != 0] ** 2)

    diag_vals_t = E_t.sum(axis=1).A1
    diag_vals_t[diag_vals_t != 0] = 1 / (diag_vals_t[diag_vals_t != 0] ** 2)

    D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
    D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))

    A_h = E_h.transpose() @ D_h_inv @ E_h
    A_t = E_t.transpose() @ D_t_inv @ E_t
    A = A_h + A_t

    return igraph.Graph.Weighted_Adjacency(A)


def get_relation_triplets(G_rel, B):
    rel_triplets = []
    for h, t in G_rel.get_edgelist():
        w = G_rel.es[G_rel.get_eid(h, t)]["weight"]
        rel_triplets.append((int(h), int(t), float(w)))
    rel_triplets = np.array(rel_triplets)

    nnz = len(rel_triplets)
    temp = (-rel_triplets[:, 2]).argsort()
    weight_ranks = np.empty_like(temp)
    weight_ranks[temp] = np.arange(nnz) + 1

    relation_triplets = []
    for idx, (h, t, w) in enumerate(rel_triplets):
        rk = int(math.ceil(weight_ranks[idx] / nnz * B)) - 1
        assert 0 <= rk < B
        relation_triplets.append([int(h), int(t), rk])

    return torch.tensor(np.array(relation_triplets))
