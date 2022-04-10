# GraphRNNReproducability
Reproducing GraphRNN paper

## Train
```
nohup python -u train.py &> train.out&
```

```
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_MLP --graph_type grid &> logs/train_MLP_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_RNN --graph_type grid &> logs/train_RNN_grid.out&
nohup python -u train.py --note GraphRNN_MLP_bfs_max --graph_type grid &> logs/train_MLP_bfs_grid.out&
nohup python -u train.py --note GraphRNN_RNN_bfs_max --graph_type grid &> logs/train_RNN_bfs_grid.out&


CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_MLP --graph_type community4 &> logs/train_MLP_community4.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_RNN --graph_type community4 &> logs/train_RNN_community4.out&

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_MLP --graph_type citeseer &> logs/train_MLP_citeseer.out&
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_RNN --graph_type citeseer &> logs/train_RNN_citeseer.out&

CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_MLP --graph_type DD &> logs/train_MLP_DD.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_RNN --graph_type DD &> logs/train_RNN_DD.out&
```

```
python train.py --note GraphRNN_RNN --graph_type loop
nohup python -u train.py --note GraphRNN_MLP --graph_type loop &> logs/train_MLP_loop.out&
nohup python -u train.py --note GraphRNN_RNN --graph_type loop &> logs/train_RNN_loop.out&
nohup python -u train.py --note GraphRNN_RNN_bfs_max --graph_type loop &> logs/train_RNN_bfs_loop.out&

```

## Evaluate
```
CUDA_VISIBLE_DEVICES=3 python evaluate.py
```
