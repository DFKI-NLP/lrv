import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from helpers.numeric import tensor_to_list


class XLinear(nn.Linear):
    """
    Code heavily based on (MIT License)
    https://github.com/Hey1Li/Salient-Relevance-Propagation/blob/master/VGG_FACE.ipynb
    accessed on 2019-02-21.
    """

    def __init__(self, linear):
        super(XLinear, self).__init__(in_features=linear.in_features,
                                      out_features=linear.out_features,
                                      bias=linear.bias)  # super(nn.Linear, self).__init__() in notebook
        self.bias = linear.bias
        if self.bias is not None:
            raise NotImplementedError
        self.weight = linear.weight
        self.__explainable = False
        self.X = None

    def relprop_plus(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V, 0, 1)) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        return R

    @staticmethod
    def all_positive(t):
        # check if all coefficients positive
        return True if torch.sum(t[t <= 0]).item() == 0 else False

    def relprop(self, R):  # todo beta is overwritten
        assert self.__explainable, 'Relprop invoked but not in explainable mode.'
        assert self.X is not None, 'Relprop invoked but no prior forward pass.'
        assert XLinear.all_positive(self.X), 'Activations negative'
        assert XLinear.all_positive(R), 'Relevance vector contains negative coefficients'
        if self.bias:
            raise NotImplementedError
        R_plus = self.relprop_plus(R)

        # for safety reasons forget cached input
        self.set_explainable(False)
        if not torch.allclose(torch.sum(R_plus), torch.sum(R)):
            diff = torch.abs(torch.sum(R_plus) - torch.sum(R)).item()
            warnings.warn('Conservation property violated with difference {}'.format(diff))

        assert XLinear.all_positive(R_plus), 'Relevance vector contains negative coefficients'

        return R_plus

    def forward(self, x):
        # if in explainable mode, cache inputs for LRP
        if self.__explainable:
            self.X = x
        return super().forward(x)

    def set_explainable(self, explainable: bool):
        # for safety reasons, forget previous inputs
        self.X = None
        self.__explainable = explainable


class XFirstLinear(nn.Linear):

    def __init__(self, linear):
        super(XFirstLinear, self).__init__(in_features=linear.in_features,
                                           out_features=linear.out_features,
                                           bias=linear.bias)
        self.bias = linear.bias
        if self.bias is not None:
            raise NotImplementedError
        self.weight = linear.weight
        self.__explainable = False
        self.X = None

    def forward(self, x):
        # if in explainable mode, cache inputs for LRP
        if self.__explainable:
            self.X = x
        return super().forward(x)

    def set_explainable(self, explainable: bool):
        # for safety reasons, forget previous inputs
        self.X = None
        self.__explainable = explainable

    def relprop(self, R, lower_bound, higher_bound):
        assert (XLinear.all_positive(R))
        # transpose because nn.Linear returns W^T
        W = torch.transpose(self.weight, 0, 1)
        V = torch.clamp(W, min=0)  # T
        U = torch.clamp(W, max=0)  # T
        X = self.X
        L = self.X * 0 + lower_bound
        H = self.X * 0 + higher_bound

        Z = torch.mm(self.X, W)
        Z = Z - torch.mm(L, V)
        Z = Z - torch.mm(H, U)
        Z = Z + 1e-9

        S = R / Z

        R_res = X * torch.mm(S, torch.transpose(W, 0, 1))
        R_res = R_res - (L * torch.mm(S, torch.transpose(V, 0, 1)))
        R_res = R_res - (H * torch.mm(S, torch.transpose(U, 0, 1)))

        assert torch.allclose(torch.sum(R), torch.sum(R_res)), 'Relevance lost.'
        assert XLinear.all_positive(R_res), 'Relevance negative.'

        return R_res


class XReLU(nn.ReLU):

    def __init__(self):
        super(XReLU, self).__init__()

    @staticmethod
    def relprop(R):
        return R


class XGraphConvolution(Module):
    """ Basic GCN layer taken from (MIT License)
    https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    on 21 February 2019. Extended w/ (MIT License)
    https://github.com/Hey1Li/Salient-Relevance-Propagation/blob/master/VGG_FACE.ipynb
    also accessed on 21 February 2019.
    """

    def __init__(self, features_in, features_out, first_layer, bias=None):
        super(XGraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_in, features_out))
        self.bias = bias
        if self.bias is not None:
            raise NotImplementedError
        self.reset_parameters()
        self.features_in = features_in
        self.features_out = features_out
        self.__explainable = False
        self.first_layer = first_layer
        self.xfc = None
        self.adj_xfc = None
        self.X = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # forget previous forward passes
        self.set_explainable(False)

    def forward(self, input, adj, sanity_checks=True):

        # if in explainable mode, use dedicated explainable FC layers and cache inputs for LRP, expect singular inputs
        if self.__explainable:

            # cache input for LRP
            self.X = input  # todo: this is only used to check if a forward pass was performed

            # an early sanity check
            assert (input.shape[0] == adj.shape[0] and adj.shape[0] == adj.shape[1])

            # prepare the explainable fully connected feature layer
            # todo: transposed weights are passed, do we need to switch in and out dimensions?
            # todo: remove assertion when sure that feature order makes sense
            assert self.features_in == self.features_out, 'In features not equal to out features'
            if not self.first_layer:
                self.xfc = XLinear(
                    nn.Linear(in_features=self.features_in, out_features=self.features_out, bias=self.bias))
            else:
                self.xfc = XFirstLinear(nn.Linear(in_features=self.features_in,
                                                  out_features=self.features_out,
                                                  bias=self.bias))
            self.xfc.set_explainable(True)
            weight_t = torch.t(self.weight)
            # PyTorch computes XW^T in nn.Linear, thus to compute XW we must pass W^T
            self.xfc.weight.data = weight_t

            # prepare the explainable fully connected layer
            self.adj_xfc = XLinear(nn.Linear(in_features=adj.shape[0],
                                             out_features=adj.shape[0],
                                             bias=self.bias))
            self.adj_xfc.set_explainable(True)
            self.adj_xfc.weight.data = adj

            # compute HW w/ explainable fully connected feature layer
            support = self.xfc(input)

            # apply relu here s.t. adjacency layer only receives positive inputs
            support = torch.nn.functional.relu(support)

            # sanity checks are expensive which is why they can be turned off
            if sanity_checks:
                # compute HW as in normal mode (see else-branch)
                support_ref = torch.mm(input, self.weight)
                support_ref = torch.nn.functional.relu(support_ref)
                # compare w/ numerical imprecision tolerance
                assert torch.allclose(support, support_ref), 'Feature maps in explainable and normal mode differ.'

            # compute A(HW) as ((HW)^TA^T)^T, recall that nn.Linear computes XW^T
            support = torch.t(support)
            output = self.adj_xfc(support)
            output = torch.t(output)

            if sanity_checks:
                output_ref = torch.mm(adj, support_ref)
                assert torch.allclose(output, output_ref), 'Fusions in explainable and normal mode differ.'

            if self.bias is not None:
                raise NotImplementedError

            return output

        # in normal mode, do not cache inputs, expect batches
        else:
            shape = input.shape
            input = input.view(shape[0] * shape[1], -1)  # (bz * #n, d)
            support = torch.mm(input, self.weight)
            support = torch.nn.functional.relu(
                support)  # apply relu here s.t. adjacency fully connected layer only receives positive inputs
            support = support.view(shape[0], shape[1], -1)
            output = torch.bmm(adj, support)  # (bz, #n, d)
            if self.bias is not None:
                raise NotImplementedError
            else:
                return output

    def relprop(self, R, lower_bound=None, higher_bound=None, sanity_checks=True):
        assert self.__explainable, 'Relprop invoked but not in explainable mode.'
        assert self.X is not None, 'Relprop invoked but no prior forward pass.'
        if self.first_layer:
            assert lower_bound is not None, 'First XLayer but no lower bound.'
            assert higher_bound is not None, 'First XLayer but no upper bound.'
            assert isinstance(self.xfc, XFirstLinear)
        if self.bias:
            raise NotImplementedError
        # The following lines compute (((R^T A^T)^T)W^T^T) = (AR)W
        R_T = torch.t(R)
        R_T_A = self.adj_xfc.relprop(R_T)
        R_A = torch.t(R_T_A)
        R_A_W = self.xfc.relprop(R_A) if not self.first_layer else self.xfc.relprop(R_A,
                                                                                    lower_bound=lower_bound,
                                                                                    higher_bound=higher_bound)
        if sanity_checks:
            # todo: this sanity check seems to be too strict
            if not torch.allclose(torch.sum(R_A, dim=1), torch.sum(R_A_W, dim=1)):
                diff_tensor = torch.sum(R_A, dim=1) - torch.sum(R_A_W, dim=1)
                diff_tensor_abs = torch.abs(diff_tensor)
                diff_scalar = diff_tensor_abs[torch.argmax(diff_tensor_abs)]
                warning = "torch.allclose failed with a maximum difference of {} " \
                          "between the two tensors.".format(diff_scalar)
                warnings.warn(message=warning)

        # for safety reasons, forget everything
        self.set_explainable(False)

        # return the relevance scores for the feature and the fusion forward pass
        return R_A_W, R_A

    def set_explainable(self, explainable):
        # for safety reasons, forget all previous forward passes
        self.X = None
        self.adj_xfc = None
        self.xfc = None
        self.__explainable = explainable

    def __repr__(self):
        return self.__class__.__name__ + '(features_in=' + str(self.features_in) + \
               ', features_out=' + str(self.features_out) + \
               ', bias=' + str(self.bias) + \
               ', explainable=' + str(self.__explainable) + \
               ')'


class XMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size):
        super(XMaxPool2d, self).__init__(kernel_size=kernel_size)
        self.__explainable = False
        self.X = None

    def set_explainable(self, explainable):
        # for safety reasons forget previous forward passes
        self.X = None
        self.__explainable = explainable

    def forward(self, x):
        if self.__explainable:
            self.X = x
            # in explainable mode, expect single data points, adjust dimensions accordingly
            self.X = self.X.squeeze(0)
            self.X = self.X.squeeze(0)
        return super().forward(x)

    @staticmethod
    def project_R(t, R):
        """
        >>> t = torch.rand(5,3) # 5 nodes, 3 features
        >>> tensor([[0.7062, 0.9978, 0.5424],
        >>>         [0.8933, 0.6349, 0.6535],
        >>>         [0.4860, 0.8061, 0.4894],
        >>>         [0.6340, 0.2327, 0.5752],
        >>>         [0.7192, 0.7298, 0.0139]])
        >>> R = torch.tensor(np.array([[.5, .3, .7]]))
        >>> tensor([0.5000, 0.3000, 0.7000], dtype=torch.float64)
        >>> project_R(t, R)
        >>> tensor([[0.0000, 0.3000, 0.0000],
        >>>         [0.5000, 0.0000, 0.7000],
        >>>         [0.0000, 0.0000, 0.0000],
        >>>         [0.0000, 0.0000, 0.0000],
        >>>         [0.0000, 0.0000, 0.0000]])
        """
        assert (len(R.shape) == 2)  # (1,d)
        assert (len(t.shape) == 2)  # (n,d)
        assert (R.shape[1] == t.shape[1])  # number of max activations equal to number of filters
        _, indices = t.max(0)  # get indices of max activations in column vectors
        res = torch.zeros_like(t)  # prepare result vector
        for col, row in enumerate(indices):  # note: col, row instead of row col!
            res[row][col] = R[0][col]  # project relevances onto sparse matrix
        return res

    def relprop(self, R):
        # todo channel, batch?
        R = self.project_R(self.X, R)
        self.set_explainable(False)
        return R


class XGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, pad, bias, dropout=.5):
        super(XGCN, self).__init__()
        self.bias = bias
        if bias is not None:
            raise NotImplementedError
        self.xgc1 = XGraphConvolution(nfeat, nhid, first_layer=True, bias=self.bias)
        self.xgc2 = XGraphConvolution(nhid, nhid, first_layer=False, bias=self.bias)
        self.dropout = dropout
        self.xmaxpool2d = XMaxPool2d(kernel_size=(pad, 1))
        self.xfc = XLinear(nn.Linear(nhid, nclass, bias=self.bias))
        self.__explainable = False

    def forward(self, embedding, adjacency):
        embedding = self.xgc1(embedding, adjacency)
        embedding = F.dropout(embedding, self.dropout, training=self.training)
        embedding = self.xgc2(embedding, adjacency)
        embedding = F.dropout(embedding, self.dropout, training=self.training)
        if self.__explainable:
            embedding = embedding.unsqueeze(0)  # batch
            embedding = embedding.unsqueeze(0)  # channel
        else:
            embedding = embedding.unsqueeze(1)  # add channel
        embedding = self.xmaxpool2d(embedding)
        embedding = embedding.squeeze(1)  # get rid of batch
        embedding = embedding.squeeze(1)  # get rid of channel
        embedding = self.xfc(embedding)
        if not self.__explainable:
            embedding = F.log_softmax(embedding, dim=1)
        return embedding

    def relprop(self, R, lower_bound, higher_bound, sanity_checks=True):
        if sanity_checks:
            warnings.warn('Sanity checks enabled, this is computationally expensive.')
        else:
            warnings.warn('Sanity checks disabled.')
        assert self.__explainable, 'Relprop invoked but not in explainable mode.'
        if self.bias:
            raise NotImplementedError
        # perform LRP and cache relevance tensors after each layer
        Rs = dict()
        Rs['R_after_out_layer'] = tensor_to_list(R)
        R = self.xfc.relprop(R)
        Rs['R_after_last_fc'] = tensor_to_list(R)
        R = self.xmaxpool2d.relprop(R)
        Rs['R_after_max_pooling_layer'] = tensor_to_list(R)
        R, R_adj2 = self.xgc2.relprop(R)
        Rs['R_after_gcn_2_adjacency_fc'] = tensor_to_list(R_adj2)
        Rs['R_after_gcn_2_feature_fc'] = tensor_to_list(R)
        R, R_adj1 = self.xgc1.relprop(R, lower_bound=lower_bound, higher_bound=higher_bound)
        Rs['R_after_gcn_1_adjacency_fc'] = tensor_to_list(R_adj1)
        Rs['R_after_gcn_1_feature_fc'] = tensor_to_list(R)
        self.set_explainable(False)
        if sanity_checks:
            # check that conversation property holds
            R1 = torch.sum(R).item()
            R2 = torch.sum(torch.tensor(np.array(Rs['R_after_out_layer']))).item()
            diff = abs(R1 - R2)
            if diff >= 1e-8:
                warnings.warn('Conservation property violated with a difference of {}'.format(diff))
        return R, Rs

    def set_explainable(self, explainable):
        if explainable:
            # disable dropout in explainable mode
            self.eval()
        else:
            self.train()
        self.xgc1.set_explainable(explainable)
        self.xgc2.set_explainable(explainable)
        self.xmaxpool2d.set_explainable(explainable)
        self.xfc.set_explainable(explainable)
        self.__explainable = explainable
