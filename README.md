# GraphRNNReproducability
Reproducing GraphRNN paper

## Train
```
nohup python -u train.py &> train.out&
```

### On multiple datsets
```
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_MLP --graph_type grid &> logs/train_MLP_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_RNN --graph_type grid &> logs/train_RNN_grid.out&

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_MLP --graph_type community4 &> logs/train_MLP_community4.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_RNN --graph_type community4 &> logs/train_RNN_community4.out&

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_MLP --graph_type citeseer &> logs/train_MLP_citeseer.out&
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_RNN --graph_type citeseer &> logs/train_RNN_citeseer.out&

CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_MLP --graph_type DD &> logs/train_MLP_DD.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_RNN --graph_type DD &> logs/train_RNN_DD.out&
```

### Different BFS
```
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_MLP_bfs_min --graph_type grid &> logs/train_MLP_min_grid.out&
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_RNN_bfs_min --graph_type grid &> logs/train_RNN_min_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_MLP_bfs_ran --graph_type grid &> logs/train_MLP_ran_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_RNN_bfs_ran --graph_type grid &> logs/train_RNN_ran_grid.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_MLP_bfs_max --graph_type grid &> logs/train_MLP_max_grid.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_RNN_bfs_max --graph_type grid &> logs/train_RNN_max_grid.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_MLP_bfs_no --graph_type grid &> logs/train_MLP_no_grid.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_RNN_bfs_no --graph_type grid &> logs/train_RNN_no_grid.out&
```

## Evaluate
```
python evaluate.py
```

