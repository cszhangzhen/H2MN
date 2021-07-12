import torch
import math
import os
import json
import numpy as np
from texttable import Texttable
from torch.utils.data import random_split, Dataset
from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, dense_to_sparse
from torch_geometric.utils import softmax, degree, sort_edge_index
from torch_scatter import scatter
from torch_cluster import random_walk
from torch_sparse import spspmm, coalesce


class BinaryFuncDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.dir_name = os.path.join(root, name)
        self.number_features = 0
        self.func2graph = dict()
        super(BinaryFuncDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.func2graph, self.number_features = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        for filename in os.listdir(self.dir_name):
            if filename[-3:] == 'npy':
                continue
            f = open(self.dir_name + '/' + filename, 'r')
            contents = f.readlines()
            f.close()
            for jsline in contents:
                check_dict = dict()
                g = json.loads(jsline)
                funcname = g['fname']  # Type: str
                features = g['features']  # Type: list
                idlist = g['succs']  # Type: list
                n_num = g['n_num']  # Type: int
                # Build graph index
                edge_index = []
                for i in range(n_num):
                    idx = idlist[i]
                    if len(idx) == 0:
                        continue
                    for j in idx:
                        if (i, j) not in check_dict:
                            check_dict[(i,j)] = 1
                            edge_index.append((i,j))
                        if (j, i) not in check_dict:
                            check_dict[(j,i)] = 1
                            edge_index.append((j,i))
                np_edge_index = np.array(edge_index).T
                pt_edge_index = torch.from_numpy(np_edge_index)
                x = np.array(features, dtype=np.float32)
                x = torch.from_numpy(x)
                row, col = pt_edge_index
                cat_row_col = torch.cat((row,col))
                n_nodes = torch.unique(cat_row_col).size(0)
                if n_nodes != x.size(0):
                    continue
                self.number_features = x.size(1)
                pt_edge_index, _ = sort_edge_index(pt_edge_index, num_nodes=x.size(0))
                data = Data(x=x, edge_index=pt_edge_index)
                data.num_nodes = n_num
                if funcname in self.func2graph:
                    self.func2graph[funcname].append(data)
                else:
                    self.func2graph[funcname] = [data]
        torch.save((self.func2graph, self.number_features), self.processed_paths[0])


class GraphClassificationDataset(object):
    def __init__(self, args):
        self.args = args
        self.training_funcs = dict()
        self.validation_funcs = dict()
        self.testing_funcs = dict()
        self.number_features = None
        self.id2name = None
        self.func2graph = None
        self.process_dataset()
    
    def process_dataset(self):
        print('\nPreparing datasets.\n')
        self.dataset = BinaryFuncDataset('datasets/{}'.format(self.args.dataset), self.args.dataset)
        self.number_features = self.dataset.number_features
        self.func2graph = self.dataset.func2graph
        self.id2name = dict()

        cnt = 0
        for k,v in self.func2graph.items():
            self.id2name[cnt] = k
            cnt += 1

        self.train_num = int(len(self.func2graph) * 0.8)
        self.val_num = int(len(self.func2graph) * 0.1)
        self.test_num = int(len(self.func2graph)) - (self.train_num + self.val_num)

        random_idx = np.random.permutation(len(self.func2graph))
        self.train_idx = random_idx[0: self.train_num]
        self.val_idx = random_idx[self.train_num: self.train_num + self.val_num]
        self.test_idx = random_idx[self.train_num + self.val_num:]

        self.training_funcs = self.split_dataset(self.training_funcs, self.train_idx)
        self.validation_funcs = self.split_dataset(self.validation_funcs, self.val_idx)
        self.testing_funcs = self.split_dataset(self.testing_funcs, self.test_idx)
    
    def split_dataset(self, funcdict, idx):
        for i in idx:
            funcname = self.id2name[i]
            funcdict[funcname] = self.func2graph[funcname]
        return funcdict

    def collate(self, data_list):
        batchS = Batch.from_data_list([data[0] for data in data_list] + [data[0] for data in data_list])
        batchT = Batch.from_data_list([data[1] for data in data_list] + [data[2] for data in data_list])
        batchL = ([1 for data in data_list] + [0 for data in data_list])
        return batchS, batchT, batchL
    
    def create_batches(self, funcs, collate, shuffle_batch=True):
        data = FuncDataset(funcs)
        loader = torch.utils.data.DataLoader(data, batch_size=self.args.batch_size, shuffle=shuffle_batch, collate_fn=collate, num_workers=8)

        return loader

    def transform(self, data):
        new_data = dict()

        new_data['g1'] = data[0].to(self.args.device)
        new_data['g2'] = data[1].to(self.args.device)
        new_data['target'] = torch.from_numpy(np.array(data[2], dtype=np.float32)).to(self.args.device)
        return new_data
    

class FuncDataset(Dataset):
    def __init__(self, funcdict):
        super(FuncDataset, self).__init__()
        self.funcdict = funcdict
        self.id2key = dict()
        cnt = 0
        for k, v in self.funcdict.items():
            self.id2key[cnt] = k
            cnt += 1
    
    def __len__(self):
        return len(self.funcdict)

    def __getitem__(self, idx):
        graphset = self.funcdict[self.id2key[idx]]
        pos_idx = np.random.choice(range(len(graphset)), size=2, replace=True)
        origin_graph = graphset[pos_idx[0]]
        pos_graph = graphset[pos_idx[1]]
        all_keys = list(self.funcdict.keys())
        neg_key = np.random.choice(range(len(all_keys)))
        while all_keys[neg_key] == self.id2key[idx]:
            neg_key = np.random.choice(range(len(all_keys)))
        neg_data = self.funcdict[all_keys[neg_key]]
        neg_idx = np.random.choice(range(len(neg_data)))
        neg_graph = neg_data[neg_idx]
        
        return origin_graph, pos_graph, neg_graph, 1, 0


class GraphRegressionDataset(object):
    def __init__(self, args):
        self.args = args
        self.training_graphs = None
        self.training_set = None
        self.val_set = None
        self.testing_graphs = None
        self.nged_matrix = None
        self.real_data_size = None
        self.number_features = None
        self.process_dataset()

    def process_dataset(self):
        print('\nPreparing dataset.\n')
        self.training_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)
        self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False)

        self.nged_matrix = self.training_graphs.norm_ged
        self.real_data_size = self.nged_matrix.size(0)
        
        max_degree = 0
        for g in self.training_graphs + self.testing_graphs:
            if g.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
        one_hot_degree = OneHotDegree(max_degree, cat=True)
        self.training_graphs.transform = one_hot_degree
        self.testing_graphs.transform = one_hot_degree

        self.number_features = self.training_graphs.num_features

        train_num = len(self.training_graphs) - len(self.testing_graphs)
        val_num = len(self.testing_graphs)
        self.training_set, self.val_set = random_split(self.training_graphs, [train_num, val_num])

    def create_batches(self, graphs):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data['g1'] = data[0].to(self.args.device)
        new_data['g2'] = data[1].to(self.args.device)

        norm_ged = self.nged_matrix[data[0]['i'].reshape(-1).tolist(), data[1]['i'].reshape(-1).tolist()].tolist()
        new_data['target'] = torch.from_numpy(np.exp([(-el) for el in norm_ged])).view(-1).float().to(self.args.device)
        return new_data


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    t.add_rows([['Parameter', 'Value']] + [[k.replace('_', ' ').capitalize(), args[k]] for k in keys])
    print(t.draw())


def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]


def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps


def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
        fid = finds[0][0]
        while fid > 0:
            cid = sort_id_mat[fid]
            pid = sort_id_mat[fid - 1]
            if data[pid] == data[cid]:
                fid -= 1
            else:
                break
        rank[i] = fid + 1
    
    return rank


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation


def hypergraph_construction(edge_index, edge_attr, num_nodes, k=2, mode='RW'):
    if mode == 'RW':
        # Utilize random walk to construct hypergraph
        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walk = random_walk(row, col, start, walk_length=k)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
        adj[walk[start], start.unsqueeze(1)] = 1.0
        edge_index, _ = dense_to_sparse(adj)
    else:
        # Utilize neighborhood to construct hypergraph
        if k == 1:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
        else:
            neighbor_augment = TwoHopNeighbor()
            hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
            hop_data.num_nodes = num_nodes
            for _ in range(k-1):
                hop_data = neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    
    return edge_index, edge_attr


def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce='mean')

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce='sum')

    return hyperedge


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
