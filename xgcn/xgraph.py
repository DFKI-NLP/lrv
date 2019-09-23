import re
import warnings
from collections import OrderedDict

import networkx as nx
import numpy as np
import torch


class XNode:
    """Labelled nodes w/ unique identifiers.
    :param id: identifier (unique).
    :param label: label (shared).
    """

    def __init__(self, id, label, **kwargs):
        assert isinstance(id, int)
        assert isinstance(label, str)
        self.id = id
        self.label = label
        self.kwargs = kwargs

    def __str__(self):
        return '({},{})'.format(self.id, self.label)

    def __repr__(self):
        return '({},{})'.format(self.id, self.label)


class XGraph(nx.OrderedDiGraph):
    """Ordered Graph w/ XNodes and no initial self loops."""

    def __init__(self):
        super(XGraph, self).__init__()

    def add_node(self, n, **attr):
        assert (isinstance(n, XNode))
        super().add_node(n, **attr)

    def add_nodes_from(self, nodes, **attr):
        for n in nodes:
            assert (isinstance(n, XNode))
        super().add_nodes_from(self, nodes, **attr)

    def get_node(self, id):
        for node in self.nodes():
            if node.id == id:
                return node
        raise KeyError

    #todo: this method is not invoked when graph was deserialized
    def nodes(self):
        nodes = super().nodes()
        nodes = list(nodes)
        return nodes

    def contains_by_id(self, id):
        nodes = self.nodes()
        for node in nodes:
            if node.id == id:
                return True
        return False

    def labels(self):
        """Returns the graph's labels in a list."""
        res = [node.label for node in self.nodes()]
        return res

    def labels_2_one_hot(self):
        """Returns label-to-one-hot-encoding dictionary."""
        lbls = list(OrderedDict.fromkeys(self.labels()))
        res = {}
        for idx, label in enumerate(lbls):
            one_hot = np.zeros((1, len(lbls)))
            one_hot[0][idx] = 1.
            res[label] = one_hot
        return res

    #  TODO catch self loops

    def E(self, label2vec):
        """Returns an embedding matrix of the graph's nodes."""
        assert('<###-unk-###>') in label2vec, 'The unknown token cannot be found.'
        lbls = self.labels()
        assert (len(lbls) > 0)
        # initialize w/ first node's embedding
        res = label2vec[lbls[0]] if lbls[0] in label2vec else label2vec['<###-unk-###>']
        for label in lbls[1:]:
            if label in label2vec:
                vec = label2vec[label]
            else:
                vec = label2vec['<###-unk-###>']
            res = np.concatenate((res, vec), axis=0)  # todo bottleneck? put in list and then concatenate?
        return res

    def A(self):
        """Returns the graph's adjacency matrix as a numpy array."""
        # transpose because networkx returns row-to-column matrix, while Kipf's implementation requires column-to-row
        return np.transpose(nx.to_numpy_matrix(self))

    def A_hat(self):
        """Returns the graph's adjacency matrix w/ self loops; see paper."""
        a = self.A()
        i = np.identity(a.shape[0])
        a_hat = np.add(a, i)
        return a_hat

    def D_hat(self):
        """Returns the graph's degree matrix (w/ added self loops); see paper."""
        a_h = self.A_hat()
        size = a_h.shape[0]
        # iterate over rows of column-to-row edge direction
        # degree (sum) of incoming edges
        d = [np.sum(a_h[i]) for i in range(size)]
        res = np.zeros((size, size))
        for idx in range(size):
            res[idx][idx] = d[idx]
        return res

    def D_tilde(self):
        """Returns D_hat^(-1/2); see paper."""
        d = self.D_hat()
        for idx in range(len(d)):
            d[idx][idx] = 1. / np.power(d[idx][idx], .5)
        return d

    def A_tilde(self):
        """Returns D_tilde * A_hat * D_tilde; see paper."""
        a_h = self.A_hat()
        d_tilde = self.D_tilde()
        a_tilde = np.matmul(np.matmul(d_tilde, a_h), d_tilde)
        a_tilde = np.asarray(a_tilde) # todo redundant?
        return a_tilde

    def to_json(self):
        res = dict()
        res['nodes'] = [{'id': node.id, 'label': node.label} for node in self.nodes()]
        res['edges'] = [{'source': edge[0].id, 'target': edge[1].id, 'type': edge[2]['t']}
                        for edge in self.edges(data=True)]
        sorted(res['edges'], key=lambda e: e['source'])
        return res


class XSample:
    def __init__(self, embedding, adjacency, label=None):
        self.EMBEDDING = embedding
        self.ADJACENCY = adjacency
        self.LABEL = label

    def to_tensor(self):
        self.EMBEDDING = torch.from_numpy(self.EMBEDDING).float()
        self.ADJACENCY = torch.from_numpy(self.ADJACENCY).float()


class Pad:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, xsample):
        E, A, L = xsample.EMBEDDING, xsample.ADJACENCY, xsample.LABEL
        assert (E.shape[0] == A.shape[0])

        if self.padding == E.shape[0]:
            pass
        elif self.padding > E.shape[0]:
            shape = (1, E.shape[1])
            for n in range(self.padding - E.shape[0]):
                zero = np.zeros(shape)
                E = np.concatenate((E, zero), axis=0)
        else:
            raise Exception("Padding must not be less than the number of nodes, but it is: {} < {}"
                            .format(self.padding, E.shape[0]))
        if self.padding == A.shape[0]:
            pass
        elif self.padding > A.shape[0]:
            padded = np.zeros((self.padding, self.padding))
            padded[:A.shape[0], :A.shape[1]] = A
            A = padded
        else:
            raise Exception("Padding must not be less than the number of nodes, but it is: {} < {}"
                            .format(self.padding, A.shape[0]))

        return XSample(E, A, L)


class ToTensor():
    def __call__(self, xsample):
        xsample.to_tensor()
        return xsample


class LabelToOneHot:
    def __init__(self, classes):
        self.classes = classes

    def label_to_one_hot(self, label):
        if label not in self.classes:
            raise KeyError
        idx = self.classes.index(label)
        return idx # this is only a scalar, which is handled by the framework (code review)

    def __call__(self, xsample):
        E, A, L = xsample.EMBEDDING, xsample.ADJACENCY, self.label_to_one_hot(xsample.LABEL)
        return XSample(E, A, L)

