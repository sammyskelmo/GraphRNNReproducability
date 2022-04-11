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
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_MLP_bfs_min --graph_type grid --bfs_mode bfs_min_deg &> logs/train_MLP_min_grid.out&
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --note GraphRNN_RNN_bfs_min --graph_type grid --bfs_mode bfs_min_deg &> logs/train_RNN_min_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_MLP_bfs_ran --graph_type grid --bfs_mode bfs_random &> logs/train_MLP_ran_grid.out&
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --note GraphRNN_RNN_bfs_ran --graph_type grid --bfs_mode bfs_random &> logs/train_RNN_ran_grid.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_MLP_bfs_max --graph_type grid --bfs_mode bfs_max_deg &> logs/train_MLP_max_grid.out&
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --note GraphRNN_RNN_bfs_max --graph_type grid --bfs_mode bfs_max_deg &> logs/train_RNN_max_grid.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_MLP_bfs_no --graph_type grid --bfs_mode no_bfs &> logs/train_MLP_no_grid.out&
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --note GraphRNN_RNN_bfs_no --graph_type grid --bfs_mode no_bfs &> logs/train_RNN_no_grid.out&
```

## Evaluate
Download graphs from https://drive.google.com/file/d/1JMijswX57XR7yxQbA5PR9tfHZ-PpnfQt/view?usp=sharing
```
python evaluate.py
nohup python -u evaluate.py &> logs/evaluate.out&

```
## Bibliography

As mentioned in our group report, some  portions of our code base were sourced from the official GitHub repository of the paper. This can be found at the following link: https://github.com/snap-stanford/GraphRNN. We also gained fundamental technical understanding through a wide body of existing literature, as cited in our written paper that was submitted for examination. 

