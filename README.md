# SAZRL

SAZRL is a knowledge graph completion model for zero-shot relation learning. It automatically constructs relation representations from a support subgraph using a graph attention network over relation co-occurrence structure.

## Requirements

```
torch
numpy
scipy
igraph
networkx
tqdm
```

## Data

Place datasets under `Data/`. Supported datasets（Data.zip, Link: https://pan.baidu.com/s/16rfumEzmaPvCeabOpCds8w?pwd=kiqf Code: kiqf）:

- `NELL-ZS`
- `Wiki-ZS`
- `FB-RZS`

Each dataset directory should contain `train.txt`, `valid.txt`, `test.txt`, and candidate JSON files. 

## Usage

**Train:**

```bash
python train.py --data_name FB-RZS --AutoRelation True
```

**Test (load checkpoint):**

```bash
python test.py --data_name Wiki-ZS
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_name` | `FB-RZS` | Dataset name |
| `--AutoRelation` | `False` | Enable relation graph encoder |
| `--dim_ent` | `200` | Entity embedding dimension |
| `--dim_rel` | `300` | Relation embedding dimension |
| `--num_layer_rel` | `3` | Number of relation encoder layers |
| `--B` | `3` | Number of edge weight bins |
| `--loss` | `soft` | Loss function (`margin`/`adv`/`crr`/`soft`/`sig`) |
| `--lr` | `5e-3` | Learning rate |
| `--num_epoch` | `50` | Training epochs |
| `--n_hop` | `2` | Subgraph sampling hops |
| `--write` | `False` | Save model checkpoints |

## Files

| File | Description |
|---|---|
| `model.py` | SAZRL model and RelationLayer |
| `train.py` | Training loop |
| `test.py` | Evaluation on saved checkpoint |
| `dataset.py` | Data loading and preprocessing |
| `evaluation.py` | Link prediction evaluation |
| `generate_G.py` | Relation graph construction |
| `utils.py` | Utilities (sampling, batching, metrics) |
| `loss.py` | Loss functions |
| `Parser.py` | Argument parsing |
