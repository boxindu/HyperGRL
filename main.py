import sys
import pickle
from sklearn.model_selection import train_test_split
import networkx as nx
import dgl
from sklearn.utils import shuffle
import torch as th
from timeit import default_timer as timer
import argparse
import nxmetis
from sklearn.metrics import accuracy_score
import itertools
from tqdm import tqdm
import os.path
from os import path

sys.path.append("./graph_scorers")

import pandas as pd
# pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)
import numpy as np
import hyperedge_clique
import hyperedge_tree
from collections import Counter


# Training settings
parser = argparse.ArgumentParser()
# Training parameter
parser.add_argument('--alpha', type=float, default=1, help='alpha for joint training')
parser.add_argument('--beta', type=float, default=1, help='beta for joint training')
parser.add_argument('--cluster_num', type=int, default=10, help='number of clusters when using dis2cluster')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--n_epochs', type=int, default=40,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--n_negative', type=int, default=8,
                    help='number of negative examples in node-level self-supervised training.')
parser.add_argument('--dataset', default="cora", help="The dataset name. 'cora', 'pubmed'")
parser.add_argument('--data_path', default="./datasets_sample/noisy/", help="The data path.")

# Model parameter
parser.add_argument('--model_type', default= 'hyperedge_clique',
                    help="Choose the model to be trained.('hyperedge_clique', 'hyperedge_tree')")
parser.add_argument('--feat_dim', type=int, default=16, help='feature dimension')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument("--train_mode", default="pretrain_n", help="The training mode. 'separate', 'joint', 'pretrain_n', "
                                                               "'pretrain_e', 'pretrain_n_e'")

args = parser.parse_args()


def code_entry(data_path, model_to_use, select_training_mode, pretrain_model, eval_metric,
               rw_node_feat=False, ori_node_feat=False, pretrain_he=False, joint=False, pretrain_node=False):
    with open(data_path + 'hyperedges.p', 'rb') as pickle_file:
        hyperedges = pickle.load(pickle_file)
    pickle_file.close()
    with open(data_path + 'node_ids.p', 'rb') as pickle_file:
        node_id = pickle.load(pickle_file)
    pickle_file.close()

    if rw_node_feat and not ori_node_feat:
        with open(data_path + 'rw_node_feat.p', 'rb') as pickle_file:
            node_feats = pickle.load(pickle_file)
        pickle_file.close()
        feature_dim = args.feat_dim
        text_feats = node_feats

    elif ori_node_feat and not rw_node_feat:
        feature_dim = args.feat_dim
        with open(data_path + 'text_feat.p', 'rb') as pickle_file:
            text_feats = pickle.load(pickle_file)
        pickle_file.close

    else:
        print("Please specify one node feature type!")

    membership, g_hyperedge = getPretrainData(hyperedges, pretrain_model, cluster_number=args.cluster_num,
                                              pretext_classification=True)
    X, Y, Y_pre, num_categories = getData(hyperedges, text_feats, node_id, model_to_use, membership, g_hyperedge,
                                          args.cluster_num, ori_feat=ori_node_feat, rw_feat=rw_node_feat)
    X_t_v, X_test, Y_t_v, Y_test, Y_pre_t_v, Y_pre_test = train_test_split(X, Y, Y_pre, train_size=0.8, test_size=0.2)
    X_train, X_vali, Y_train, Y_vali, Y_pre_train, Y_pre_vali = train_test_split(X_t_v, Y_t_v, Y_pre_t_v,
                                                                                 train_size=0.75, test_size=0.25)

    Y_train_counter = Counter(Y_train)
    print('Hyperedge classification training data statistics:', Y_train_counter)
    weights_tune_loss = []
    for i in range(num_categories):
        weights_tune_loss.append(Y_train_counter[i])
    weights_tune_loss = [1 - (weight / sum(weights_tune_loss)) for weight in weights_tune_loss]

    Y_train_counter = Counter(Y_pre_train)
    print('Self-training clustering statistics:', Y_train_counter)
    weights_pre_loss = []
    for i in range(len(Y_train_counter)):
        weights_pre_loss.append(Y_train_counter[i])
    weights_pre_loss = [1 - (weight / sum(weights_pre_loss)) for weight in weights_pre_loss]

    if model_to_use == 'hyperedge_clique':
        scorer = hyperedge_clique.HyperScorerGeneral(input_dim=feature_dim, hidden_size=args.hidden_dim, num_class=num_categories,
                                                     n_epochs=args.n_epochs, weight_tune=weights_tune_loss,
                                                     cluster_num=args.cluster_num, weight_pre=weights_pre_loss,
                                                     n_negative=args.n_negative, lr=args.lr)
    if model_to_use == 'hyperedge_tree':
        scorer = hyperedge_tree.HyperScorerGeneral(input_dim=feature_dim, hidden_size=args.hidden_dim, num_class=num_categories,
                                                     n_epochs=args.n_epochs, weight_tune=weights_tune_loss,
                                                   cluster_num=args.cluster_num, weight_pre=weights_pre_loss,
                                                   n_negative=args.n_negative, lr=args.lr)

    # Train graph scorer
    print("Training graph scorer")
    start_train_clustrer = timer()
    if model_to_use == 'hyperedge_clique' or model_to_use == 'hyperedge_tree':
        scorer.train(g_list_train=X_train, labels_train_tune=Y_train, labels_train_pre=Y_pre_train, g_list_validation=X_vali,
                 labels_validation_tune=Y_vali, labels_validation_pre=Y_pre_vali, eval_metric=eval_metric, alpha=1,
                 train_mode=select_training_mode, pretrain_node=pretrain_node, pretrain_he=pretrain_he, joint=joint)

    print('FINISHED training graph scorer in', round(timer() - start_train_clustrer, 2), 'seconds')

    # Performance on validation set:
    start_test_performance = timer()
    predicted_labels = scorer.predict_labels(X_vali)
    Y_vali = np.array(Y_vali)
    vali_label_list = []
    predict_label_list = []
    for i in range(Y_vali.shape[0]):
        if len(np.where(predicted_labels[i] == 1)[0]) > 0:
            predict_label_list.append(np.where(predicted_labels[i] == 1)[0][0])
        else:
            predict_label_list.append(-1)
        vali_label_list.append(Y_vali[i])

    validation_score = eval_metric(vali_label_list, predict_label_list)
    print('vali score: ', validation_score)
    print("FINISHED computing performance on validation set in", round(timer() - start_test_performance, 4), 'seconds')

    # Performance on test set:
    start_test_performance = timer()
    predicted_labels = scorer.predict_labels(X_test)
    Y_test = np.array(Y_test)
    test_label_list = []
    predict_label_list = []
    for i in range(Y_test.shape[0]):
        if len(np.where(predicted_labels[i] == 1)[0]) > 0:
            predict_label_list.append(np.where(predicted_labels[i] == 1)[0][0])
        else:
            predict_label_list.append(-1)
        test_label_list.append(Y_test[i])
    test_score = eval_metric(test_label_list, predict_label_list)
    naive_score = eval_metric(test_label_list, np.zeros(len(predict_label_list)).tolist())
    print('test score: ', test_score)
    print('naive score: ', naive_score)
    print("FINISHED computing performance on test set in", round(timer() - start_test_performance, 4), 'seconds')

    return test_score

def getPretrainData(hyperedges, pretrain_task, cluster_number=20, pretext_classification=True):
    if pretrain_task == 'dis2cluster':
        hyperedge_list = []
        for h1 in tqdm(hyperedges):
            for h2 in hyperedges:
                overlap_nodes = list(set(hyperedges[h1]["members"]) & set(hyperedges[h2]["members"]))
                if h1 != h2 and len(overlap_nodes) > 0:
                    hyperedge_list.append((h1, h2, len(overlap_nodes)))
        node_list = list(hyperedges.keys())
        g_hyperedge = nx.Graph()
        g_hyperedge.add_nodes_from(node_list)
        g_hyperedge.add_weighted_edges_from(hyperedge_list)
        (st, parts) = nxmetis.partition(g_hyperedge, cluster_number)
        if pretext_classification:
            cluster_membership = {node: membership for node, membership in enumerate(parts)}
            reversed_membership = {}
            for key in cluster_membership:
                for member in cluster_membership[key]:
                    if member not in reversed_membership:
                        reversed_membership[member] = key
            with open('./g_hyperedge.p', 'wb') as fp:
                pickle.dump(g_hyperedge, fp)
            with open('./reversed_membership.p', 'wb') as fp:
                pickle.dump(reversed_membership, fp)
            return reversed_membership, g_hyperedge
        else: # TODO: output the dict of hyperedges with distance to the centroids
            pass
    elif pretrain_task == 'dis2hyperedges':
        pass # TODO
    else:
        print('Wrong pretext_task type!')
        return True


def getData(hyperedges, node_features, node_ids, model_to_use, cluster_membership, g_hyperedge, cluster_num, rw_feat=False, ori_feat=False):
    # method 1: clique expansion. Output: list of cliques (dgl graphs) and list of labels
    if model_to_use == 'hyperedge_clique' or model_to_use == 'hyperedge_tree':
        i = 0
        lists = []
        labels_pre = []
        labels = []
        label_set = set()
        for h in hyperedges:
            if h in list(g_hyperedge.nodes):
                label_set.add(hyperedges[h]["category"])
        num_categories = len(label_set)
        for h in hyperedges:
            if h in list(g_hyperedge.nodes):
                vertex_embedding_list = []
                hyperedge = hyperedges[h]
                g_nx = nx.complete_graph(len(hyperedge["members"]))
                for vertex in hyperedge["members"]:
                    i += 1
                    if i % 100000 == 0:
                        print(i)
                    try:
                        if rw_feat:
                            vertex_embedding_list.append(th.tensor(node_features[node_ids.index(vertex)].tolist()))
                        if ori_feat:
                            vertex_embedding_list.append(th.tensor(node_features[vertex].tolist()))
                    except:
                        print("Missed one: ", vertex)
                node_attr_dict = dict(zip(list(range(len(hyperedge["members"]))), vertex_embedding_list))
                nx.set_node_attributes(g_nx, node_attr_dict, 'node_attr')
                # g = dgl.DGLGraph()
                # g.from_networkx(g_nx, node_attrs=['node_attr']) # For dgl < 0.5.x
                g = dgl.from_networkx(g_nx, node_attrs=['node_attr']) # For dgl 0.5.x
                lists.append(g)

                ## Use categorical labels
                labels.append(int(hyperedge["category"]) - 1)
                # labels.append(int(hyperedge["category"]))
                labels_pre.append(int(cluster_membership[h]))

        X, Y, Y_pre = shuffle(lists, labels, labels_pre)
        return X, Y, Y_pre, num_categories
    else:
        print('Wrong model name!')
        return


if __name__ == '__main__':
    data_path_root = args.data_path
    dataset_name = args.dataset

    ##----------- Current model names: "hyperedge_clique", "hyperedge_tree"----------##
    pretrain_model = 'dis2cluster'
    pretrain_model_variants = []
    select_models_to_use = args.model_type

    ##----------- Current training mode: 'separate', 'pretrain', 'joint'------------##
    eval_metric = accuracy_score
    data_path = data_path_root + dataset_name + '/'
    final_score = 0
    num_iter = 3
    vali_cumacc = []
    vali_cumloss = []

    for i in range(num_iter):
        if args.train_mode == 'pretrain_e':
            test_score = code_entry(data_path, select_models_to_use, 'pretrain', pretrain_model, eval_metric,
                                ori_node_feat=True, pretrain_he=True, joint=False, pretrain_node=False)
        if args.train_mode == 'pretrain_n':
            test_score = code_entry(data_path, select_models_to_use, 'pretrain', pretrain_model, eval_metric,
                                                         ori_node_feat=True, pretrain_he=False, joint=False,
                                                         pretrain_node=True)
        if args.train_mode == 'pretrain_n_e':
            test_score = code_entry(data_path, select_models_to_use, 'pretrain', pretrain_model, eval_metric,
                                                         ori_node_feat=True, pretrain_he=True, joint=False,
                                                         pretrain_node=True)
        if args.train_mode == 'joint':
            test_score = code_entry(data_path, select_models_to_use, 'joint', pretrain_model, eval_metric,
                                                         ori_node_feat=True, pretrain_he=False, joint=True,
                                                         pretrain_node=False)
        if args.train_mode == 'separate':
            test_score = code_entry(data_path, select_models_to_use, 'separate', pretrain_model, eval_metric,
                                                         ori_node_feat=True, pretrain_he=False, joint=False,
                                                         pretrain_node=False)
        final_score += test_score
    final_score = final_score / num_iter
    print('Final score: ', final_score)