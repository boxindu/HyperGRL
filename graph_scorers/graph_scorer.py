from abc import ABCMeta, abstractmethod


class GraphScorer(object):
    """TODO: description"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def weight_init(self):
        """"TODO: description"""
        pass

    @abstractmethod
    def train(self, g_train, labels_train, g_validation, labels_validation, edge_features, eval_metric,
              edge_label_available, alpha):
        """"TODO: description"""
        pass

    @abstractmethod
    def predict_proba(self, g, edge_label_available):
        """"TODO: description"""
        pass

    @abstractmethod
    def predict_labels(self, g, threshold):
        """"TODO: description"""
        pass





