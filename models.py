import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv

from layers import ReadoutModule, MLPModule, CrossGraphConvolution, HyperedgeConv, HyperedgePool
from utils import hypergraph_construction


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.k = args.k
        self.mode = args.mode

        self.num_features = args.num_features

        self.conv0 = GCNConv(self.num_features, self.nhid)
        self.conv1 = HypergraphConv(self.nhid, self.nhid)
        self.cross_conv1 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool1 = HyperedgePool(self.nhid, self.args.ratio1)
        
        self.conv2 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv2 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool2 = HyperedgePool(self.nhid, self.args.ratio2)
        
        self.conv3 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv3 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool3 = HyperedgePool(self.nhid, self.args.ratio3)

        self.readout0 = ReadoutModule(self.args)
        self.readout1 = ReadoutModule(self.args)
        self.readout2 = ReadoutModule(self.args)
        self.readout3 = ReadoutModule(self.args)

        self.mlp = MLPModule(self.args)

    def forward(self, data):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        
        edge_attr_1 = data['g1'].edge_attr
        edge_attr_2 = data['g2'].edge_attr

        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
        # Layer 0
        # Graph Convolution Operation
        f1_conv0 = F.leaky_relu(self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv0 = F.leaky_relu(self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2)

        att_f1_conv0 = self.readout0(f1_conv0, batch_1)
        att_f2_conv0 = self.readout0(f2_conv0, batch_2)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)

        edge_index_1, edge_attr_1 = hypergraph_construction(edge_index_1, edge_attr_1, num_nodes=features_1.size(0), k=self.k, mode=self.mode)
        edge_index_2, edge_attr_2 = hypergraph_construction(edge_index_2, edge_attr_2, num_nodes=features_2.size(0), k=self.k, mode=self.mode)
        
        # Layer 1
        # Hypergraph Convolution Operation
        f1_conv1 = F.leaky_relu(self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv1 = F.leaky_relu(self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2)
        
        # Hyperedge Pooling
        edge1_conv1, edge1_index_pool1, edge1_attr_pool1, edge1_batch_pool1 = self.pool1(f1_conv1, batch_1, edge_index_1, edge_attr_1)
        edge2_conv1, edge2_index_pool1, edge2_attr_pool1, edge2_batch_pool1 = self.pool1(f2_conv1, batch_2, edge_index_2, edge_attr_2)
        
        # Cross Graph Convolution
        hyperedge1_cross_conv1, hyperedge2_cross_conv1 = self.cross_conv1(edge1_conv1, edge1_batch_pool1, edge2_conv1, edge2_batch_pool1)

        # Readout Module
        att_f1_conv1 = self.readout1(hyperedge1_cross_conv1, edge1_batch_pool1)
        att_f2_conv1 = self.readout1(hyperedge2_cross_conv1, edge2_batch_pool1)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)

        # Layer 2
        # Hypergraph Convolution Operation
        f1_conv2 = F.leaky_relu(self.conv2(hyperedge1_cross_conv1, edge1_index_pool1, edge1_attr_pool1), negative_slope=0.2)
        f2_conv2 = F.leaky_relu(self.conv2(hyperedge2_cross_conv1, edge2_index_pool1, edge2_attr_pool1), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv2, edge1_index_pool2, edge1_attr_pool2, edge1_batch_pool2 = self.pool2(f1_conv2, edge1_batch_pool1, edge1_index_pool1, edge1_attr_pool1)
        edge2_conv2, edge2_index_pool2, edge2_attr_pool2, edge2_batch_pool2 = self.pool2(f2_conv2, edge2_batch_pool1, edge2_index_pool1, edge2_attr_pool1)
        
        # Cross Graph Convolution
        hyperedge1_cross_conv2, hyperedge2_cross_conv2 = self.cross_conv2(edge1_conv2, edge1_batch_pool2, edge2_conv2, edge2_batch_pool2)

        # Readout Module
        att_f1_conv2 = self.readout2(hyperedge1_cross_conv2, edge1_batch_pool2)
        att_f2_conv2 = self.readout2(hyperedge2_cross_conv2, edge2_batch_pool2)
        score2 = torch.cat([att_f1_conv2, att_f2_conv2], dim=1)

        # Layer 3
        # Hypergraph Convolution Operation
        f1_conv3 = F.leaky_relu(self.conv3(hyperedge1_cross_conv2, edge1_index_pool2, edge1_attr_pool2), negative_slope=0.2)
        f2_conv3 = F.leaky_relu(self.conv3(hyperedge2_cross_conv2, edge2_index_pool2, edge2_attr_pool2), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv3, edge1_index_pool3, edge1_attr_pool3, edge1_batch_pool3 = self.pool3(f1_conv3, edge1_batch_pool2, edge1_index_pool2, edge1_attr_pool2)
        edge2_conv3, edge2_index_pool3, edge2_attr_pool3, edge2_batch_pool3 = self.pool3(f2_conv3, edge2_batch_pool2, edge2_index_pool2, edge2_attr_pool2)

        # Cross Graph Convolution
        hyperedge1_cross_conv3, hyperedge2_cross_conv3 = self.cross_conv3(edge1_conv3, edge1_batch_pool3, edge2_conv3, edge2_batch_pool3)

        # Readout Module
        att_f1_conv3 = self.readout3(hyperedge1_cross_conv3, edge1_batch_pool3)
        att_f2_conv3 = self.readout3(hyperedge2_cross_conv3, edge2_batch_pool3)
        score3 = torch.cat([att_f1_conv3, att_f2_conv3], dim=1)

        scores = torch.cat([score0, score1, score2, score3], dim=1)
        scores = self.mlp(scores)

        return scores
