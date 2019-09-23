import argparse
import copy
import ujson
import os
import pickle
import re
import warnings

import numpy as np
# noinspection PyUnresolvedReferences
import scispacy
import spacy
import torch
from tqdm import tqdm

from helpers.config import config
from helpers.log import log
from helpers.numeric import tensor_to_list, my_round
from preprocess import line_to_graph
from train import PubMedDataset
from xgcn.xgcn import XGCN
from xgcn.xgraph import Pad, XSample, XGraph


class Explanation:
    """Holds an explanation, for a forward pass: the relevance flow, the true label and the predicted label."""

    def __init__(self, graph, relevances, true_label=None, predicted_label=None, relevances_prior=None):
        self.graph: XGraph = graph
        self.relevances: list[float] = relevances
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.relevances_prior = relevances_prior


def relevance_matrix(graph, relevances_prior, relevances_now, edge_weights=None):
    """Computes the amount of relevance mass that each edge carried during LRP and writes it into an adjacency matrix."""

    # this matrix will contain the result
    if edge_weights is None:
        edge_weights = np.zeros((len(graph.nodes()), len(graph.nodes())))

    # collect leaves
    leaves = [x for x in graph.nodes() if graph.out_degree(x) == 0 and graph.in_degree(x) == 1]

    # if there are no leaves anymore, there are only roots (spacy sometimes contains multiple roots)
    if len(leaves) == 0:
        return edge_weights

    for leaf in leaves:
        relevance_diff = relevances_prior[leaf.id - 1] - relevances_now[leaf.id - 1]  # compute missing relevance
        edges = graph.in_edges(leaf)  # this should contain one edge, over which the relevance must have travelled
        assert len(edges) == 1, 'More than one incoming edge.'
        edge = list(edges)[0]  # the one and only incoming edge
        relevances_prior[edge[0].id - 1] += relevance_diff  # move relevance to parent
        # note: row -> col direction correct here, since adjacency matrix not touched
        edge_weights[(edge[0].id - 1)][(edge[1].id - 1)] = relevance_diff
        graph.remove_node(leaf)  # remove leaf (and edges)
    return relevance_matrix(graph, relevances_prior, relevances_now, edge_weights)


def normalize_explanations(graph,
                           relevance_flow,
                           true_label = None,
                           predicted_label = None):
    """Normalizes relevance flow layerwise into Explanation objects."""

    explanations = []

    # for sanity checks, collect the input relevance tensor
    R_total = np.sum(relevance_flow['R_after_out_layer'])

    # for visualization collect relevance maps after the second gcn, after the first gcn and in the input space
    R_max, R_2, R_1 = np.asarray(relevance_flow['R_after_max_pooling_layer']), \
                      np.asarray(relevance_flow['R_after_gcn_2_feature_fc']), \
                      np.asarray(relevance_flow['R_after_gcn_1_feature_fc'])

    # sum over the nodes in the graph (tokens in the sequence)
    relevances_max = np.sum(R_max, axis=1)
    relevances_2 = np.sum(R_2, axis=1)
    relevances_1 = np.sum(R_1, axis=1)

    # discard pad nodes, but check that they contain no relevance
    relevances_nodes_max = relevances_max[:len(graph.nodes())]
    relevances_nodes_2 = relevances_2[:len(graph.nodes())]
    relevances_nodes_1 = relevances_1[:len(graph.nodes())]

    if not np.allclose(np.sum(relevances_nodes_max), R_total):
        diff = np.abs(np.sum(relevances_nodes_max) - R_total)
        warnings.warn('Relevance mass of {} contained in padded nodes.'.format(diff))

    if not np.allclose(np.sum(relevances_nodes_2), R_total):
        diff = np.abs(np.sum(relevances_nodes_2) - R_total)
        warnings.warn('Relevance mass of {} contained in padded nodes.'.format(diff))

    if not np.allclose(np.sum(relevances_nodes_1), R_total):
        diff = np.abs(np.sum(relevances_nodes_1) - R_total)
        warnings.warn('Relevance mass of {} contained in padded nodes.'.format(diff))

    # normalize the relevance scores
    relevances_nodes_max = relevances_nodes_max / np.sum(relevances_nodes_max)
    relevances_nodes_2 = relevances_nodes_2 / np.sum(relevances_nodes_2)
    relevances_nodes_1 = relevances_nodes_1 / np.sum(relevances_nodes_1)

    # round, but be sure not to lose too much relevance in the process
    relevances_nodes_rounded_max = my_round(4, relevances_nodes_max)
    relevances_nodes_rounded_2 = my_round(4, relevances_nodes_2)
    relevances_nodes_rounded_1 = my_round(4, relevances_nodes_1)
    assert np.allclose(np.sum(relevances_nodes_max), 1.0), 'Normalized relevance does not sum up to one.'
    assert np.allclose(np.sum(relevances_nodes_2), 1.0), 'Normalized relevance does not sum up to one.'
    assert np.allclose(np.sum(relevances_nodes_1), 1.0), 'Normalized relevance does not sum up to one.'

    # wrap in Explanation class
    explanation_max: Explanation = Explanation(graph=graph,
                                               relevances=relevances_nodes_rounded_max,
                                               true_label=true_label,
                                               predicted_label=predicted_label)
    explanations.append(explanation_max)

    explanation_2: Explanation = Explanation(graph=graph,
                                             relevances=relevances_nodes_rounded_2,
                                             relevances_prior=relevances_nodes_max,
                                             true_label=true_label,
                                             predicted_label=predicted_label)
    explanations.append(explanation_2)

    explanation_1: Explanation = Explanation(graph=graph,
                                             relevances=relevances_nodes_rounded_1,
                                             relevances_prior=relevances_nodes_rounded_2,
                                             true_label=true_label,
                                             predicted_label=predicted_label)
    explanations.append(explanation_1)

    return graph, explanations


def padmat(mat, padding, zeros=True):
    """Pads 2-dim matrix mat along both dimensions."""

    assert len(mat.shape) == 2, 'Only defined for 2-dim matrices.'
    if padding > mat.shape[0]:
        padded = np.zeros((padding, padding)) if zeros else np.ones((padding, padding))
        padded[:mat.shape[0], :mat.shape[1]] = mat
        mat = padded
        return mat
    if padding == mat.shape[0]:
        return mat
    else:
        raise Exception("Padding smaller than matrix.")


def get_mask(matrix, relevances_and_positions, percentage, top=True):
    """Returns a mask for matrix with the top or bottom k edges masked."""
    mask = np.ones_like(matrix)
    _drop = int(len(relevances_and_positions) * percentage)
    if not top:
        relevances_and_positions = list(reversed(relevances_and_positions))
    masked_weights = relevances_and_positions[:_drop]
    for _, position in masked_weights:
        mask[position[0]][position[1]] = 0.0
    return mask, _drop


def occlude(graph, jsonl, xgcn, adjacency, embedding, drop, step, padding, verbose, line_counter):
    """Masks most and least relevant edges and tests model performance with masked adjancency matrix."""
    xgcn.eval()
    xgcn.set_explainable(True)

    CLASSES = PubMedDataset.classes()
    relevance_flow = jsonl['relevance_flow']

    # normalize relevances layerwise, save in Explanation objects
    graph, explanations = normalize_explanations(graph, relevance_flow)

    # for the first and second layer, determine how much relevance mass was carried by each edge during LRP
    def explanations_to_relevance_matrices(explanations):
        for explanation in explanations:
            if explanation.relevances_prior is not None:
                graph = copy.deepcopy(explanation.graph)
                rel_matrix = relevance_matrix(graph=graph,
                                        relevances_prior=explanation.relevances_prior,
                                        relevances_now=explanation.relevances)
                yield rel_matrix

    # note: relevance matrices in row->col edge direction
    relevance_matrices = explanations_to_relevance_matrices(explanations)
    relevance_matrices = list(relevance_matrices)
    assert len(relevance_matrices) == 2, 'Sanity check failed.'

    # normalize relevance matrices along layer dimensions, note: in row-col direction
    global_normalized_relevance_matrix = (relevance_matrices[0] + relevance_matrices[1]) / np.sum(
        np.sum(relevance_matrices[0] + relevance_matrices[1]))
    if not (np.isclose(np.sum(np.sum(global_normalized_relevance_matrix)), 1.0)):
        warnings.warn(f"After normalization sum of weights not close to 1 in line {line_counter}.")

    # now retrieve the edge relevances and their positions in the adjacency matrix, note: row->col edge direction
    edge_relevances_and_positions = []
    for edge in graph.edges:
        position = (edge[0].id - 1, edge[1].id - 1)
        weight = global_normalized_relevance_matrix[position[0]][position[1]]
        edge_relevances_and_positions.append((weight, position))

    # the following are the edges - their positions - ordered by the relevance they carried, note: row->col direction
    edge_weights_and_positions = sorted(edge_relevances_and_positions, key=lambda tup: tup[0], reverse=True)

    # now occlude and recored performance
    last_drop = None
    for ratio in np.arange(0., (drop + step), step=step):
        assert 0 <= ratio <= 1.0, 'Ratio out of range.'

        # mask for top k edges, note: in row->col direction
        mask_top, dropped_edges = get_mask(matrix=global_normalized_relevance_matrix,
                                           relevances_and_positions=edge_weights_and_positions,
                                           percentage=ratio)

        # do not repeat experiment with same number of edges dropped
        if last_drop is not None and last_drop == dropped_edges:
            continue
        else:
            last_drop = dropped_edges
        if verbose:
            log(f"Dropped {dropped_edges} weights of {len(edge_weights_and_positions)} at ratio {ratio}")

        mask_top = padmat(mask_top, padding, zeros=False)
        # since adjacency matrix in col->row direction, transpose mask, which is currently in row->col direction
        mask_top = np.transpose(mask_top)
        mask_top = torch.from_numpy(mask_top)

        # mask for bottom k edges, note: in row->col direction
        mask_bottom, droped_edges_bottom = get_mask(matrix=global_normalized_relevance_matrix,
                                                    relevances_and_positions=edge_weights_and_positions,
                                                    percentage=ratio,
                                                    top=False)

        mask_bottom = padmat(mask_bottom, padding, zeros=False)
        mask_bottom = np.transpose(mask_bottom)
        mask_bottom = torch.from_numpy(mask_bottom)

        assert (adjacency.size() == mask_top.size())
        assert (adjacency.size() == mask_bottom.size())

        # get new (masked) adjacency matrices
        a_masked_top = torch.mul(adjacency.double(), mask_top)
        a_masked_bottom = torch.mul(adjacency.double(), mask_bottom)

        # perform forward pass with new masked adjancency matrices
        pred_tensor_top = xgcn(embedding=embedding, adjacency=a_masked_top.float())
        pred_tensor_bottom = xgcn(embedding=embedding, adjacency=a_masked_bottom.float())

        # determine predicted labels
        max_idx_top = pred_tensor_top.argmax()
        max_idx_bottom = pred_tensor_bottom.argmax()
        pred_label_top = CLASSES[max_idx_top.item()]
        pred_label_bottom = CLASSES[max_idx_bottom.item()]

        # sanity check: if nothing was occluded this should be the same prediction as in the original forward pass
        if ratio == 0:
            assert pred_label_top == jsonl['prediction']['label'], "Different labels but drop ratio 0.0."
            assert pred_label_bottom == jsonl['prediction']['label'], "Different labels but drop ratio 0.0."

        # save everything in the json line
        if 'occlusion' not in jsonl:
            jsonl['occlusion'] = {}

        jsonl['occlusion'][str(ratio)] = {}
        jsonl['occlusion'][str(ratio)]['dropped_edges'] = dropped_edges
        jsonl['occlusion'][str(ratio)]['top'] = {}
        jsonl['occlusion'][str(ratio)]['bottom'] = {}
        jsonl['occlusion'][str(ratio)]['top']['label'] = pred_label_top
        jsonl['occlusion'][str(ratio)]['bottom']['label'] = pred_label_bottom
        jsonl['occlusion'][str(ratio)]['top']['tensor'] = tensor_to_list(pred_tensor_top)
        jsonl['occlusion'][str(ratio)]['bottom']['tensor'] = tensor_to_list(pred_tensor_bottom)
    return jsonl


def explain(nfeat,
            nhid,
            padding,
            path_model,
            path_text,
            path_out,
            path_label2vec,
            lower_bound,
            upper_bound,
            language_model,
            to_lower,
            crop,
            do_occlude,
            drop=None,
            step=None,
            verbose=True):
    if do_occlude:
        assert drop is not None, 'Define drop range.'
        assert step is not None, 'Define step size.'
        # assert 0 < drop <= 1 - step, 'Drop range or step size outside of valid scope.'
    if crop > 0:
        warnings.warn("Cropping dataset.")

    assert lower_bound <= upper_bound, 'Lower bound greater than upper bound'

    CLASSES = PubMedDataset.classes()
    nclasses = len(CLASSES)

    # declare model
    xgcn = XGCN(nfeat, nhid, nclasses, padding, None)

    # load weights, assume model was trained on a GPU but when loading, map to current location
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    xgcn.load_state_dict(torch.load(path_model, map_location=device))

    # pubmed data specific pattern used to identify lines that contain no data point
    pattern = "###[0-9]+$"
    pattern = re.compile(pattern)

    # spacy creates the dependency tree
    nlp = spacy.load(language_model)

    # label2vec dictionary, used to map node labels (i.e. tokens) onto embeddings
    label2vec = pickle.load(open(path_label2vec, 'rb'))

    # all graph embeddings need to be of the same size
    pad = Pad(padding=padding)

    with open(path_text, 'r') as fin:

        # lines processed
        line_counter = 0
        # each json line is self contained at the price of redundancy
        with open(path_out, 'w+') as fout:
            for line in tqdm(fin.readlines()):

                # if line contains no data point, skip
                if pattern.match(line) or len(line.strip()) == 0:
                    continue

                line_counter = line_counter + 1
                if 0 <= crop <= line_counter:
                    log(f"Terminating after {line_counter} lines, due to crop of {crop}.")
                    break

                # disable dropouts
                xgcn.eval()

                # cache inputs during forward pass
                xgcn.set_explainable(True)

                # declare json line
                jsonl = dict()

                jsonl['line'] = line_counter

                # save config
                jsonl['padding'] = padding

                # save vocabulary path
                jsonl['label2vec'] = path_label2vec

                # save model state
                jsonl['model'] = {}
                jsonl['model']['path'] = path_model
                jsonl['model']['device'] = device
                jsonl['model']['architecture'] = xgcn.__repr__().strip().replace(os.linesep, '')

                # save raw text
                jsonl['text'] = line.strip()

                # retrieve label and graph, save
                label, graph = line_to_graph(line=line, nlp=nlp, to_lower=to_lower)
                graph_json = graph.to_json()
                jsonl['graph'] = graph_json
                jsonl['label'] = label

                # perform forward pass, save resulting tensor
                jsonl['prediction'] = dict()
                e, a = graph.E(label2vec=label2vec), graph.A_tilde()
                x_sample = XSample(embedding=e, adjacency=a)
                x_sample = pad(x_sample)
                x_sample.to_tensor()
                e, a = x_sample.EMBEDDING, x_sample.ADJACENCY

                # since xgcn is in explainable mode, softmax layer should be deactivated
                pred_tensor = xgcn(embedding=e, adjacency=a)
                jsonl['prediction']['tensor'] = tensor_to_list(pred_tensor)

                # save label
                max_idx = pred_tensor.argmax()

                if pred_tensor[0][max_idx] <= 0:
                    warnings.warn(f'Maximum output of GCN is <=0 (line {line_counter}), will ignore this data point.')
                    continue

                pred_label = CLASSES[max_idx.item()]
                jsonl['prediction']['label'] = pred_label

                # perform layerwise relevance propagation, save layerwise relevance
                R = torch.zeros_like(pred_tensor)
                R[0][max_idx] = pred_tensor[0][max_idx]
                _, relevance_flow = xgcn.relprop(R,
                                                 lower_bound=lower_bound,
                                                 higher_bound=upper_bound)
                jsonl['relevance_flow'] = relevance_flow

                if do_occlude:
                    jsonl = occlude(graph=graph,
                                    jsonl=jsonl,
                                    xgcn=xgcn,
                                    adjacency=a,
                                    embedding=e,
                                    drop=drop,
                                    step=step,
                                    padding=padding,
                                    verbose=verbose,
                                    line_counter=line_counter)

                fout.write((ujson.dumps(jsonl) + os.linesep))
                fout.flush()
            fout.close()

    return True

if __name__ == '__main__':
    log('Explaining...')
    cfg = config('./config.json')

    print(ujson.dumps(cfg, indent=2))

    parser = argparse.ArgumentParser()

    parser.add_argument('--nfeat', type=int, default=cfg['training']['nfeat'])
    parser.add_argument('--nhid', type=int, default=cfg['training']['nhid'])
    parser.add_argument('--path_model', type=str, default=cfg['training']['path_model'])
    parser.add_argument('--pad', type=int, default=cfg['training']['pad'])
    parser.add_argument('--file_test_text', type=str, default=cfg['preprocessing']['pubmed']['file_test_text'])
    parser.add_argument('--vocab', type=str, default=cfg['preprocessing']['word_vectors']['vocab'])
    parser.add_argument('--lower_bound', type=float, default=cfg['preprocessing']['word_vectors']['lower_bound'])
    parser.add_argument('--upper_bound', type=float, default=cfg['preprocessing']['word_vectors']['upper_bound'])
    parser.add_argument('--file_explanations_jsonl', type=str, default=cfg['explain']['file_explanations_jsonl'])
    parser.add_argument('--to_lower', type=bool, default=cfg['preprocessing']['pubmed']['to_lower'])
    parser.add_argument('--language_model', type=str, default=cfg['preprocessing']['pubmed']['language_model'])
    parser.add_argument('--crop', type=int, default=cfg['explain']['crop'])
    parser.add_argument('--occlude', type=bool, default=cfg['explain']['occlude'])
    parser.add_argument('--drop', type=float, default=cfg['explain']['drop'])
    parser.add_argument('--step', type=float, default=cfg['explain']['step'])
    parser.add_argument('--verbose', type=bool, default=cfg['explain']['verbose'])

    args = parser.parse_args()

    explain(nfeat=args.nfeat,
            nhid=args.nhid,
            path_model=args.path_model,
            padding=args.pad,
            path_text=args.file_test_text,
            path_out=args.file_explanations_jsonl,
            path_label2vec=args.vocab,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            to_lower=args.to_lower,
            language_model=args.language_model,
            crop=args.crop,
            do_occlude=args.occlude,
            drop=args.drop,
            step=args.step,
            verbose=args.verbose)

