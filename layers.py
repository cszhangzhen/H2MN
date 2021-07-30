import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_add_pool, global_mean_pool, HypergraphConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_scatter import scatter_add
from torch_scatter import scatter
from torch_geometric.utils import degree
from utils import zeros, glorot, hyperedge_representation


class HypergraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(HypergraphConvolution, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm):
        out = norm[edge_index_i].view(-1, 1) * x_j.view(-1, self.out_channels)

        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1 
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class HyperedgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(HyperedgeConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm):
        out = norm[edge_index_i].view(-1, 1) * x_j.view(-1, self.out_channels)

        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)

        num_nodes = hyperedge_index[0].max().item() + 1
        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], num_nodes, x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        out = B.view(-1, 1) * x
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, size=(num_edges, num_nodes))

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class HyperedgePool(MessagePassing):
    def __init__(self, nhid, ratio):
        super(HyperedgePool, self).__init__()
        self.ratio = ratio
        self.nhid = nhid
        self.alpha = 0.1
        self.K = 10
        self.hypergnn = HypergraphConv(self.nhid, 1)

    def message(self, x_j, edge_index_i, norm):
        out = norm[edge_index_i].view(-1, 1) * x_j.view(-1, 1)

        return out
    
    def forward(self, x, batch, edge_index, edge_weight):
        # Init pagerank values
        pr = torch.sigmoid(self.hypergnn(x, edge_index, edge_weight))

        if edge_weight is None:
            D = degree(edge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(edge_weight[edge_index[1]], edge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if edge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = edge_index[1].max().item() + 1 
        B = 1.0 / degree(edge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if edge_weight is not None:
            B = B * edge_weight

        hidden = pr
        for k in range(self.K):
            self.flow = 'source_to_target'
            out = self.propagate(edge_index, x=pr, norm=B)
            self.flow = 'target_to_source'
            pr = self.propagate(edge_index, x=out, norm=D)
            pr = pr * (1 - self.alpha)
            pr += self.alpha * hidden

        score = self.calc_hyperedge_score(pr, edge_index)
        score = score.view(-1)
        perm = topk(score, self.ratio, batch)
        
        x_hyperedge = hyperedge_representation(x, edge_index)
        x_hyperedge = x_hyperedge[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = self.filter_hyperedge(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x_hyperedge, edge_index, edge_attr, batch
    
    def calc_hyperedge_score(self, x, edge_index):
        x = x[edge_index[0]]
        score = scatter(x, edge_index[1], dim=0, reduce='mean')

        return score
    
    def filter_hyperedge(self, edge_index, edge_attr, perm, num_nodes):
        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        mask = (mask[col] >= 0)
        row, col = row[mask], col[mask]

        # ID re-mapping operation, which makes the ids become continuous 
        unique_row = torch.unique(row)
        unique_col = torch.unique(col)
        combined = torch.cat((unique_row, unique_col))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]

        new_perm = torch.cat((unique_col, difference))
        max_id = new_perm.max().item() + 1
        new_mask = new_perm.new_full((max_id,), -1)
        j = torch.arange(new_perm.size(0), dtype=torch.long, device=new_perm.device)
        new_mask[new_perm] = j

        row, col = new_mask[row], new_mask[col]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return torch.stack([row, col], dim=0), edge_attr


class CrossGraphConvolutionOperator(MessagePassing):
    def __init__(self, out_nhid, in_nhid):
        super(CrossGraphConvolutionOperator, self).__init__('add')
        self.out_nhid = out_nhid
        self.in_nhid = in_nhid
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_nhid, self.in_nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, assign_index, N, M):
        global_x = self.propagate(assign_index, size=(N, M), x=x)
        target_x = x[1]
        target_x = torch.unsqueeze(target_x, dim=1)
        global_x = torch.unsqueeze(global_x, dim=1)
        weight = torch.unsqueeze(self.weight, dim=0)
        target_x = target_x * weight
        global_x = global_x * weight
        numerator = torch.sum(target_x * global_x, dim=-1)
        target_x_denominator = torch.sqrt(torch.sum(torch.square(target_x), dim=-1) + 1e-6)
        global_x_denominator = torch.sqrt(torch.sum(torch.square(global_x), dim=-1) + 1e-6)
        denominator = torch.clamp(target_x_denominator * global_x_denominator, min=1e-6)

        return numerator / denominator

    def message(self, x_i, x_j, edge_index):
        x_i_norm = torch.norm(x_i, dim=-1, keepdim=True)
        x_j_norm = torch.norm(x_j, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_i_norm * x_j_norm, min=1e-6)
        x_product = torch.sum(x_i * x_j, dim=1, keepdim=True)
        coef = F.relu(x_product / x_norm)
        coef_sum = scatter(coef + 1e-6, edge_index[1], dim=0, reduce='sum')
        normalized_coef = coef / coef_sum[edge_index[1]]

        return normalized_coef * x_j


class CrossGraphConvolution(torch.nn.Module):
    def __init__(self, out_nhid, in_nhid):
        super(CrossGraphConvolution, self).__init__()
        self.out_nhid = out_nhid
        self.in_nhid = in_nhid
        self.cross_conv = CrossGraphConvolutionOperator(self.out_nhid, self.in_nhid)
    
    def forward(self, x_left, batch_left, x_right, batch_right):
        num_nodes_x_left = scatter_add(batch_left.new_ones(x_left.size(0)), batch_left, dim=0)
        shift_cum_num_nodes_x_left = torch.cat([num_nodes_x_left.new_zeros(1), num_nodes_x_left.cumsum(dim=0)[:-1]], dim=0)
        cum_num_nodes_x_left = num_nodes_x_left.cumsum(dim=0)

        num_nodes_x_right = scatter_add(batch_right.new_ones(x_right.size(0)), batch_right, dim=0)
        shift_cum_num_nodes_x_right = torch.cat([num_nodes_x_right.new_zeros(1), num_nodes_x_right.cumsum(dim=0)[:-1]], dim=0)
        cum_num_nodes_x_right = num_nodes_x_right.cumsum(dim=0)

        adj = torch.zeros((x_left.size(0), x_right.size(0)), dtype=torch.float, device=x_left.device)
        # Construct batch fully connected graph in block diagonal matirx format
        for idx_i, idx_j, idx_x, idx_y in zip(shift_cum_num_nodes_x_left, cum_num_nodes_x_left, shift_cum_num_nodes_x_right, cum_num_nodes_x_right):
            adj[idx_i:idx_j, idx_x:idx_y] = 1.0
        new_edge_index, _ = self.dense_to_sparse(adj)
        row, col = new_edge_index

        assign_index1 = torch.stack([col, row], dim=0)
        out1 = self.cross_conv((x_right, x_left), assign_index1, N=x_right.size(0), M=x_left.size(0))

        assign_index2 = torch.stack([row, col], dim=0)
        out2 = self.cross_conv((x_left, x_right), assign_index2, N=x_left.size(0), M=x_right.size(0))

        return out1, out2

    def dense_to_sparse(self, adj):
        assert adj.dim() == 2
        index = adj.nonzero(as_tuple=False).t().contiguous()
        value = adj[index[0], index[1]]
        return index, value


class ReadoutModule(torch.nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(ReadoutModule, self).__init__()
        self.args = args

        self.weight = torch.nn.Parameter(torch.Tensor(self.args.nhid, self.args.nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :param size: Size
        :return representation: A graph level representation matrix.
        """
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class MLPModule(torch.nn.Module):
    def __init__(self, args):
        super(MLPModule, self).__init__()
        self.args = args

        self.lin0 = torch.nn.Linear(self.args.nhid * 2 * 4, self.args.nhid * 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self.args.nhid * 2, self.args.nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self.args.nhid, self.args.nhid // 2)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(self.args.nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = F.relu(self.lin2(scores))
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = torch.sigmoid(self.lin3(scores)).view(-1)

        return scores
