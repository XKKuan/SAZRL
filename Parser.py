import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_path', default="Data", type=str)
    parser.add_argument('--data_name', default='FB-RZS', type=str)
    parser.add_argument('--cuda', default=7, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--n_hop', default=2, type=int)
    parser.add_argument('--margin', default=4.0, type=float)

    parser.add_argument('--AutoRelation', default=False, type=bool)
    parser.add_argument('--subgraph', default=True, type=bool)
    parser.add_argument('--B', default=3, type=int)

    parser.add_argument('--batch_test', default=1, type=int)
    parser.add_argument('--num_support', default=1, type=int)

    parser.add_argument('--num_neg', default=3000, type=int)
    parser.add_argument('--adv_temperature', default=0.2, type=float)
    parser.add_argument('--loss', default='soft', type=str)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--schedul', default=False, type=bool)

    parser.add_argument('--dim_ent', default=200, type=int)
    parser.add_argument('--dim_rel', default=300, type=int)
    parser.add_argument('--hid_dim_ratio_rel', default=2, type=int)
    parser.add_argument('--num_layer_rel', default=3, type=int)

    parser.add_argument('--n_batch', default=500, type=int)
    parser.add_argument('--candidate', default=True, type=bool)

    parser.add_argument('--valid_epochs', default=1, type=int)
    parser.add_argument('--test_epochs', default=1, type=int)

    parser.add_argument('--write', default=False, type=bool)

    return parser.parse_args()
