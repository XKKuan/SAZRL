import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(RelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2*dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)

        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim=-1, keepdim=True) + self.attn_bin[relation_triplets[..., 2]]

        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1), device=relation_triplets.device).scatter_reduce(
            dim=0, index=scatter_idx, src=attn_val_raw, reduce='amax', include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1), device=relation_triplets.device).index_add(
            dim=0, index=head_idxs, source=attn_val)

        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel), device=relation_triplets.device).index_add(
            dim=0, index=head_idxs,
            source=beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1, -1)


class SAZRL(nn.Module):
    def __init__(self, num_ent, dim_ent, num_rel, dim_rel, hid_dim_ratio_rel, num_layer_rel, num_bin, device, num_head=8, bias=True, auto=True):
        super().__init__()

        self.auto = auto
        self.bias = bias
        self.num_layer_rel = num_layer_rel
        self.act = nn.ReLU()
        layer_dim_rel = hid_dim_ratio_rel * dim_rel

        self.emb_ent = torch.nn.Embedding(num_ent, dim_ent)

        if self.auto:
            self.init_emb_rel = torch.zeros((num_rel, dim_rel), device=device)

            self.layers_rel = nn.ModuleList([
                RelationLayer(layer_dim_rel, layer_dim_rel, num_bin, bias=bias, num_head=num_head)
                for _ in range(num_layer_rel)
            ])
            self.res_proj_rel = nn.ModuleList([
                nn.Linear(layer_dim_rel, layer_dim_rel, bias=bias)
                for _ in range(num_layer_rel)
            ])

            self.rel_proj1 = nn.Linear(dim_rel, layer_dim_rel, bias=self.bias)
            self.rel_proj2 = nn.Linear(layer_dim_rel, dim_rel, bias=self.bias)
            self.rel_convert = nn.Linear(dim_rel, dim_ent, bias=self.bias)
        else:
            self.emb_rel = torch.nn.Embedding(num_rel, dim_ent)

        self.initialize()

    def initialize(self):
        gain = torch.nn.init.calculate_gain('relu')

        torch.nn.init.xavier_normal_(self.emb_ent.weight.data, gain=gain)

        if self.auto:
            torch.nn.init.xavier_normal_(self.init_emb_rel, gain=gain)

            nn.init.xavier_normal_(self.rel_proj1.weight, gain=gain)
            nn.init.xavier_normal_(self.rel_proj2.weight, gain=gain)
            nn.init.xavier_normal_(self.rel_convert.weight, gain=gain)
            for i in range(self.num_layer_rel):
                nn.init.xavier_normal_(self.res_proj_rel[i].weight, gain=gain)

            if self.bias:
                nn.init.zeros_(self.rel_proj1.bias)
                nn.init.zeros_(self.rel_proj2.bias)
                nn.init.zeros_(self.rel_convert.bias)
                for i in range(self.num_layer_rel):
                    nn.init.zeros_(self.res_proj_rel[i].bias)
        else:
            torch.nn.init.xavier_normal_(self.emb_rel.weight.data, gain=gain)

    def forward(self, relation_triplets):
        if self.auto:
            layer_emb_rel = self.rel_proj1(self.init_emb_rel)

            for i, layer in enumerate(self.layers_rel):
                layer_emb_rel = layer(layer_emb_rel, relation_triplets) + \
                                self.res_proj_rel[i](layer_emb_rel)
                layer_emb_rel = self.act(layer_emb_rel)

            return self.emb_ent, self.rel_proj2(layer_emb_rel)
        else:
            return self.emb_ent, self.emb_rel

    def get_score(self, ent_emb, rel_emb, triples):
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]

        h_emb = ent_emb(h)
        r_emb = self.rel_convert(rel_emb[r]) if self.auto else rel_emb(r)
        t_emb = ent_emb(t)

        return (h_emb * r_emb * t_emb).sum(dim=-1)
