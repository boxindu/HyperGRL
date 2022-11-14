"""Pet-GNNH with clique expansion using GAT."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from hyper_model_general import HyperGNNScorerGeneral
import gensim.downloader as api
import torch.optim as optim
from dgl.nn.pytorch.glob import Set2Set
from dgl.nn import AvgPooling
from dgl.nn import GraphConv
import dgl
import random

class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, class_num, dropout_rate = 0.5, num_heads = 5, cluster_num=20, n_negative=10):
        super(GATClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.cluster_num = cluster_num

        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, self.num_heads, feat_drop=0.5, attn_drop=0.0, activation=nn.ReLU())
        ])
        self.GCN_layer = GraphConv(in_dim, hidden_dim)
        self.pool_graph = Set2Set(hidden_dim, 1, 1)
        self.mean_pool = AvgPooling()
        self.n_negative = n_negative

        # Output a single output class
        self.classify_graph = nn.Sequential(nn.Linear(hidden_dim , int(hidden_dim/2)), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(int(hidden_dim/2), self.class_num), nn.Softmax(dim=1))

        # Use regression model in dis2cluster pretext task
        self.regression_graph = nn.Linear(hidden_dim * 2, self.cluster_num)

        # Use classification model in dis2cluster pretext task
        self.classify_graph_to_cluster = nn.Sequential(nn.Linear(hidden_dim , int(hidden_dim/2)), nn.ReLU(), nn.Dropout(0.5),
                                                       nn.Linear(int(hidden_dim/2), self.cluster_num), nn.Softmax(dim=1))
        self.distinguisher = nn.Sigmoid()

    def forward(self, g):
        g_list = dgl.unbatch(g)
        g_size = [g.number_of_nodes() for g in g_list]
        h = g.ndata['node_attr']

        ## GCN:
        # h = self.GCN_layer(g, h)
        # hg = self.mean_pool(g, h)

        ## GAT:
        for conv in self.layers:
            hi, _ = conv(g, h)
        hi = th.sum(hi, dim=1) * (1/self.num_heads)
        with g.local_scope():
            g.ndata['result'] = hi
            hg = self.mean_pool(g, g.ndata['result'])

        h_list = []
        product_list = []
        counter = 0
        for i in range(len(g_list)):
            h_list.append(h[counter:counter + g_size[i], :])
        counter = 0
        for i in range(len(g_list)):
            h_temp = h[counter:counter+g_size[i], :]
            selected_index = random.randint(0, g_size[i] - 1)
            h_selected_node = h_temp[selected_index, :]
            context_index = th.tensor(list(range(g_size[i]))) != selected_index
            h_selected_context = th.mean(h_temp[context_index], dim=0)
            product_list.append(th.dot(h_selected_context, h_selected_node))
            counter += g_size[i]
            for j in range(self.n_negative):
                rand = 0
                while rand == i:
                    rand = random.randint(0, len(g_size) - 1)
                if i != rand:
                    selected_n_index = random.randint(0, g_size[rand] - 1)
                    h_n_node = h_list[rand][selected_n_index, :]
                    product_list.append(th.dot(h_n_node, h_selected_context) / (th.norm(th.tensor(h_n_node)) *
                                                                                th.norm(th.tensor(h_selected_context))))
        products = th.tensor(product_list, requires_grad=True)
        if th.isnan(products).any():
            print('Has nan!!!')

        label_pre_node = th.zeros(self.n_negative + 1, requires_grad=False)
        label_pre_node[0] = 1
        label_pre_node = label_pre_node.repeat(len(g_list))

        return self.classify_graph(hg), self.classify_graph_to_cluster(hg), self.distinguisher(products), label_pre_node

class GATConv(nn.Module):
    r"""
    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.1,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        # self._edge_feats = edge_feats
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.fc_edge = nn.Linear(
        #     self._edge_feats, out_feats * num_heads, bias=False
        # )
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(th.FloatTensor(size=(self._edge_feats, 1)))
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        def concat_src_dst(edges):
            # return {'out_edge': th.cat((edges.src['ft'], edges.dst['ft']), dim=2)}
            # return {'out_edge': th.abs(edges.src['ft'] - edges.dst['ft'])}
            return {'out_edge': th.abs(edges.src['ori_ft'] - edges.dst['ori_ft'])}

        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = F.dropout(feat[0], self.feat_drop, training=self.training)
            h_dst = F.dropout(feat[1], self.feat_drop, training=self.training)
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = F.dropout(feat, self.feat_drop, training=self.training)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
            graph.srcdata.update({'ori_ft': feat})
            graph.dstdata.update({'ori_ft': feat})

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

        # e = self.sigmoid(graph.edata.pop('e')) # Option: Use sigmoid instead of leaky_relu plus softmax normalization
        e = self.leaky_relu(graph.edata.pop('e'))
        # # compute softmax
        graph.edata['e'] = edge_softmax(graph, e)
        graph.edata['all_one'] = th.ones_like(graph.edata['e'])  # GNN

        # message passing
        graph.update_all(fn.u_mul_e('ft', 'e', 'm'),
                         fn.sum('m', 'ft'))

        graph.update_all(fn.u_mul_e('ori_ft', 'e', 'm'), fn.sum('m', 'ori_ft'))  # update ori_ft by attention e

        rst = graph.dstdata['ft']
        # graph.apply_edges(fn.u_add_v('ft', 'ft', 'out_edge'))  # Option 1: Summation of src and dst representation as edge representation
        graph.apply_edges(func=concat_src_dst)  # Option 2: Concatenation of src and dst representation as edge representation

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst, graph.edata['out_edge']


class HyperScorerGeneral(HyperGNNScorerGeneral):
    """
    A scorer based on the HyperGNNScorerGeneral class
    """
    def __init__(self, input_dim= 0, num_class=0, thresholds = None, hidden_size=64, n_epochs=1, dropout=0.5, n_heads = 5,
                 thresholds_edge= None, cluster_num=20, batch_size=64, weight_tune=None, weight_pre=None, n_negative=10, lr=0.01):
        super(HyperGNNScorerGeneral, self).__init__()
        self._input_dim = input_dim
        self._num_class = num_class
        self._thresholds = thresholds
        self._thresholds_edge = thresholds_edge
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._n_epochs = n_epochs
        self._dropout = dropout
        self._num_heads = n_heads
        self._cluster_num = cluster_num
        self._n_negative = n_negative

        self._model = GATClassifier(
                                    self._input_dim,
                                    self._hidden_size,
                                    self._num_class,
                                    num_heads= self._num_heads,
                                    dropout_rate=self._dropout,
                                    cluster_num=cluster_num,
                                    n_negative=self._n_negative
                                    )
        if weight_tune:
            self._loss_func_tune = nn.CrossEntropyLoss(weight=th.tensor(weight_tune))
        else:
            self._loss_func_tune = nn.CrossEntropyLoss()
        if weight_pre:
            self._loss_func_pre = nn.CrossEntropyLoss(weight=th.tensor(weight_pre))
        else:
            self._loss_func_pre = nn.CrossEntropyLoss()
        self._loss_func_node = nn.BCELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, 'min', patience=3, factor=0.2, verbose=True, min_lr=10 ** -5)



