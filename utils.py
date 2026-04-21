import torch
import random
import networkx as nx
import numpy as np
from collections import deque, defaultdict


def remove_duplicate(x):
    return list(dict.fromkeys(x))


def read_KG(path):
    entity, relation, triplet = [], [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            entity.append(h)
            entity.append(t)
            relation.append(r)
            triplet.append((h, r, t))
    return remove_duplicate(entity), remove_duplicate(relation), remove_duplicate(triplet)


def sample_entity(triplet, x, thr=50, n_hop=2):
    adj = defaultdict(set)
    for h, _, t in triplet:
        adj[h].add(t)
        adj[t].add(h)

    visited = set(x)
    queue = deque([(e, 0) for e in x])
    while queue:
        node, hop = queue.popleft()
        if hop >= n_hop:
            continue
        neighbors = adj.get(node, set())
        sampled = random.sample(neighbors, min(thr, len(neighbors))) if neighbors else []
        for neighbor in sampled:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hop + 1))

    return visited


def gcc(triplets):
    G = nx.Graph()
    G.add_edges_from((h, t) for h, _, t in triplets)
    return max(nx.connected_components(G), key=len) if G.nodes else set()


def batch_generator1(triplets, p_num):
    relation_dict = {}
    for h, r, t in triplets:
        if r not in relation_dict:
            relation_dict[r] = []
        relation_dict[r].append(torch.tensor([h, r, t]))

    relations = list(relation_dict.keys())
    random.shuffle(relations)
    indexes = {r: 0 for r in relations}

    while True:
        all_done = True
        for r in relations:
            start_idx = indexes[r]
            if start_idx >= len(relation_dict[r]):
                continue
            all_done = False
            end_idx = min(start_idx + p_num, len(relation_dict[r]))
            yield torch.stack(relation_dict[r][start_idx:end_idx])
            indexes[r] = end_idx
        if all_done:
            break


def batch_generator2(triplets, p_num):
    relation_dict = {}
    for h, r, t in triplets:
        if r not in relation_dict:
            relation_dict[r] = []
        relation_dict[r].append(torch.tensor([h, r, t]))

    relations = list(relation_dict.keys())
    indexes = {r: 0 for r in relations}
    current_pointer = 0
    residual_data = []

    while True:
        random.shuffle(relations)
        all_done = True

        for _ in range(len(relations)):
            r = relations[current_pointer]
            current_pointer = (current_pointer + 1) % len(relations)

            remaining = len(relation_dict[r]) - indexes[r]
            if remaining <= 0:
                continue

            if remaining >= p_num:
                start = indexes[r]
                batch = relation_dict[r][start:start + p_num]
                indexes[r] += p_num
                yield torch.stack(batch)
                all_done = False
            else:
                start = indexes[r]
                residual_data.extend(relation_dict[r][start:])
                indexes[r] = len(relation_dict[r])

                while len(residual_data) < p_num:
                    found = False
                    for other_r in relations:
                        other_remaining = len(relation_dict[other_r]) - indexes[other_r]
                        if other_remaining > 0:
                            take_num = min(p_num - len(residual_data), other_remaining)
                            start = indexes[other_r]
                            residual_data.extend(relation_dict[other_r][start:start + take_num])
                            indexes[other_r] += take_num
                            found = True
                            break
                    if not found:
                        break

                if len(residual_data) >= p_num:
                    yield torch.stack(residual_data[:p_num])
                    residual_data = residual_data[p_num:]
                    all_done = False

        if all_done:
            if residual_data:
                yield torch.stack(residual_data)
            break


def generate_neg(triplets, data, num_neg=10):
    num_ent = data.num_ent

    neg_triplets = triplets.unsqueeze(dim=1).repeat(1, num_neg, 1)
    rand_result = torch.rand((len(triplets), num_neg), device=triplets.device)
    perturb_head = rand_result < 0.5
    perturb_tail = rand_result >= 0.5
    rand_idxs = torch.randint(low=0, high=num_ent - 1, size=(len(triplets), num_neg), device=triplets.device)
    rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:, :, 0][perturb_head]
    rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:, :, 2][perturb_tail]
    neg_triplets[:, :, 0][perturb_head] = rand_idxs[perturb_head]
    neg_triplets[:, :, 2][perturb_tail] = rand_idxs[perturb_tail]
    return torch.cat(torch.split(neg_triplets, 1, dim=1), dim=0).squeeze(dim=1)


def get_rank(triplet, scores, filters, candidates, target=2):
    true_index = candidates.index(triplet[0, target].item())
    true_value = scores[true_index].item()
    scores[[candidates.index(i) for i in filters if i != triplet[0, target].item()]] = true_value - 1
    return ((scores > true_value).sum() + 1).item()


def get_metrics(ranks):
    rank = np.array(ranks, dtype=int)
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank <= 10) / len(rank)
    hit5 = np.sum(rank <= 5) / len(rank)
    hit1 = np.sum(rank <= 1) / len(rank)
    return mr, mrr, hit10, hit5, hit1
