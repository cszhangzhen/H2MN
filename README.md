# H2MN
H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks (KDD 2021)

![](https://github.com/cszhangzhen/H2MN/blob/main/fig/model.png)

This is a PyTorch implementation of the H2MN algorithm, which reasons over a pair of graph-structured objects for graph similarity learning. Specifically, the proposed method consist of the following four steps: (1) hypergraph construction, which transforms ordinary graph into hypergraph; (2) hypergraph convolution, which learns the high-order node representations; (3) hyperedge pooling, which coarsens each graph into a coarse graph to accelerate the matching procedure; (4) subgraph matching, which conducts multi-perspective subgraph matching for similarity learning. 


## Requirements
* python3.7
* pytorch==1.6.0
* torch_geometric==1.6.1
* torch_scatter==2.0.5
* torch_sparse==0.6.7
* torch_cluster==1.5.7
* sklearn==0.23.1
* numpy==1.16.0
* scipy==1.5.0
* texttable==1.6.2

This code repository is heavily built on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which is a Geometric Deep Learning Extension Library for PyTorch. Please refer [here](https://pytorch-geometric.readthedocs.io/en/latest/) for how to install and utilize the library.

## Graph-Graph Classification Datasets
 **FFmpeg** and **OpenSSL** two datasets are generated from two popular open-source software FFmpeg and OpenSSL, in which each graph denotes the binary function’s control flow graph. In our experiment, we compile the source code function under various settings such as different compilers (e.g., gcc or clang) and different optimization levels to generate multiple binary function graphs. Thus, we take two binary functions compiled from the same source code as semantically similar to each other.

* **fname**, the function's name
* **features**, the node's feature
* **n_num**, the number of nodes
* **succs**, each node's successor nodes

More detailed information can be found [here](https://github.com/runningoat/hgmn_dataset).

## Graph-Graph Regression Datasets
**AIDS**, **LINUX** and **IMDB** are used in graph-graph regression task, where each graph represents a chemical compound, program function and ego-network, respectively. Each dataset contains the ground-truth Graph Edit Distance (GED) scores between every pair of graphs. More detailed information can be found [here](https://github.com/yunshengb/SimGNN).

## Run
Just execuate the following command for graph-graph classification task:
```
python main_classification.py --datasets openssl_min50
```

Similarly, execuate the following command for graph-graph regression task:
```
python main_regression.py --datasets AIDS700nef
```

## Citing
If you find H2MN useful for your research, please consider citing the following paper:
```
@article{zhang2021graph,
  title={H2MN: Graph Similarity Learning with Hierarchical Hypergraph Matching Networks},
  author={Zhang, Zhen and Bu, Jiajun and Ester, Martin and Li, Zhao and Yao, Chengwei and Yu, Zhi and Wang, Can},
  journal={KDD},
  year={2021}
}
``` 

