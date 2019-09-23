import copy
import os
import pickle
import ujson
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from explain import normalize_explanations, relevance_matrix
from helpers.config import config
from helpers.log import log
from train import PubMedDataset
from xgcn.xgraph import XGraph, XNode


def get_latex_header():
    latex = "\\documentclass[a4paper, landscape]{article}" + os.linesep
    latex += "\\usepackage[left=0cm, right=0cm, top=0cm]{geometry}" + os.linesep
    latex += "\\usepackage{tikz-dependency}" + os.linesep
    latex += "\\begin{document}" + os.linesep
    return latex


def get_latex_footer():
    footer = "\\end{document}" + os.linesep
    return footer


def escape(token):
    return token.replace('%', '\%')


def edge_strength(x, weight, base):
    res = round(base + (weight * abs(x)), 3)
    return str(res)


def read_explanations(path_explanations):
    with open(path_explanations, 'r') as fin:
        top = []
        bottom = []
        for line in tqdm(fin):
            jsonl = ujson.loads(line)
            label = jsonl['label']
            occlusion_experiment = jsonl['occlusion']
            for percentage in occlusion_experiment:
                top_label = occlusion_experiment[percentage]['top']['label']
                bottom_label = occlusion_experiment[percentage]['bottom']['label']
                if percentage == 0:
                    assert (top_label == bottom_label)
                top.append({'percentage': percentage, 'label': label, 'prediction': top_label})
                bottom.append({'percentage': percentage, 'label': label, 'prediction': bottom_label})
        return top, bottom


def dep_edges_to_latex(explanation, weight, base):
    edge_weights = None
    if explanation.relevances_prior is not None:
        graph = copy.deepcopy(explanation.graph)
        edge_weights = relevance_matrix(graph=graph,
                                        relevances_prior=explanation.relevances_prior,
                                        relevances_now=explanation.relevances)
    latex = ""

    for edge in explanation.graph.edges(data=True):
        if edge[0].id == 0:
            latex += "\\deproot{" + str(edge[1].id) + "}{ROOT}" + os.linesep
            continue
        if edge_weights is not None:
            latex += "\\depedge[line width=" + edge_strength(edge_weights[edge[0].id - 1][edge[1].id - 1],
                                                             weight=weight, base=base) + "pt]{" + str(
                edge[0].id) + "}{" + str(edge[1].id) + "}{" + edge[2]['t'] + "}" + os.linesep
        else:
            latex += "\\depedge{" + str(edge[0].id) + "}{" + str(edge[1].id) + "}{" + edge[2]['t'] + "}" + os.linesep
    return latex


def dep_graph_to_latex(explanation, weight, base):
    latex = "\\begin{figure}" + os.linesep
    latex += "\\begin{center}" + os.linesep
    latex += "\\begin{dependency}" + os.linesep

    latex += "\\begin{deptext}" + os.linesep
    first_token = True

    nodes = explanation.graph.nodes()

    for node in nodes:
        if node.label == '-ROOT-':
            continue
        if node.kwargs['type'] != 'TOKEN':  # TODO TOKEN flag legacy?
            latex += " \\\\" + os.linesep
            break
        else:
            relevance = explanation.relevances[node.id - 1] * 100
            color = 'red' if relevance >= 0 else 'blue'
            if not first_token:
                latex += " \& " + "|[top color={}!{}]|".format(color, abs(relevance)) + escape(
                    node.label)
            else:
                latex += "|[top color={}!{}]|".format(color, abs(relevance)) + escape(
                    node.label)
                first_token = False
    first_token = True
    latex += "\\\\" + os.linesep
    for node in nodes:
        if node.label == '-ROOT-':
            continue
        if node.kwargs['type'] != 'TOKEN':  # TODO TOKEN flag legacy?
            latex += " \\\\" + os.linesep
            break
        else:
            if not first_token:
                latex += " \& " + "({})".format(round((explanation.relevances[node.id - 1] * 100), ndigits=5))
            else:
                latex += "({})".format(round((explanation.relevances[node.id - 1] * 100), ndigits=5))  # node.label
                first_token = False
    latex += "\\\\"
    latex += os.linesep + "\\end{deptext}" + os.linesep
    latex += dep_edges_to_latex(explanation, weight=weight, base=base)
    latex += "\\end{dependency}" + os.linesep
    latex += "\\end{center}" + os.linesep
    latex += "\\caption{True Label: " + explanation.true_label + " Predicted Label: " + explanation.predicted_label + "}" + os.linesep
    latex += "\\end{figure}" + os.linesep
    latex += "\\clearpage" + os.linesep

    return latex


def explanations_to_latex(explanations, weight, base):
    latex = get_latex_header()
    for explanation in explanations:
        latex += dep_graph_to_latex(explanation, weight=weight, base=base)
    latex += get_latex_footer()
    return latex


def to_latex(path_in, path_out, weight, base, max_seq_len=-1, crop=-1):
    all_explanations = []
    with open(path_in, 'r') as fin:
        for jsonl in tqdm(fin):
            # early stopping (consider tex out-of-resources error)
            if 0 < crop < len(all_explanations):
                log('Reached max number of explanations.')
                break
            jsonl = ujson.loads(jsonl)

            nodes = jsonl['graph']['nodes']
            if max_seq_len > 0 and len(nodes) > max_seq_len:
                log(f"Skipping line because max seq length exceeded.")
                continue
            edges = jsonl['graph']['edges']

            label_true = jsonl['label']
            label_pred = jsonl['prediction']['label']

            graph = XGraph()
            for node in nodes:
                graph.add_node(XNode(id=node['id'], label=node['label'], type='TOKEN'))

            for edge in edges:
                graph.add_edge(graph.get_node(edge['source']), graph.get_node(edge['target']), t=edge['type'])

            # collect the relevance flow through the layers
            relevance_flow = jsonl['relevance_flow']

            _, explanations = normalize_explanations(graph=graph,
                                                     relevance_flow=relevance_flow,
                                                     true_label=label_true,
                                                     predicted_label=label_pred)
            all_explanations = all_explanations + explanations

    latex = explanations_to_latex(explanations=all_explanations, weight=weight, base=base)
    with open(path_out, 'w') as fout:
        fout.write(latex)


def occlusion_predictions(occlusion_experiment):
    CLASSES = PubMedDataset.classes()
    percentages = set()
    for experiment in occlusion_experiment:
        percentages.add(experiment['percentage'])
    percentages = sorted(list(percentages))
    occlusion_experiment_dict = [[[], []] for _ in range(len(percentages))]
    for experiment in occlusion_experiment:
        percentage = percentages.index(experiment['percentage'])
        occlusion_experiment_dict[percentage][0].append(CLASSES.index(experiment['label']))
        occlusion_experiment_dict[percentage][1].append(CLASSES.index(experiment['prediction']))

    percentages = [(float(percentage) * 100) for percentage in percentages]
    return occlusion_experiment_dict, percentages


if __name__ == '__main__':
    cfg = config('./config.json')

    parser = ArgumentParser()
    parser.add_argument('--do_plot_occlusion_experiment', type=bool,
                        default=cfg['postprocess']['occlusion_experiment']['doit'])
    parser.add_argument('--path_in_explanations_jsonl', type=str, default=cfg['explain']['file_explanations_jsonl'])
    parser.add_argument('--path_out_top_masked_predictions', type=str,
                        default=cfg['postprocess']['occlusion_experiment']['path_out_top_masked_predictions'])
    parser.add_argument('--path_out_bottom_masked_predictions', type=str,
                        default=cfg['postprocess']['occlusion_experiment']['path_out_bottom_masked_predictions'])
    parser.add_argument('--draw_plot', type=bool, default=cfg['postprocess']['occlusion_experiment']['draw_plot'])
    parser.add_argument('--do_convert_to_latex', type=bool, default=cfg['postprocess']['latex']['doit'])
    parser.add_argument('--path_out_latex', type=str, default=cfg['postprocess']['latex']['path_out_latex'])
    parser.add_argument('--max_seq_len', type=int, default=cfg['postprocess']['latex']['max_seq_len'])
    parser.add_argument('--weight', type=float, default=cfg['postprocess']['latex']['weight'])
    parser.add_argument('--base', type=float, default=cfg['postprocess']['latex']['base'])
    parser.add_argument('--crop', type=int, default=cfg['postprocess']['latex']['crop'])
    args = parser.parse_args()

    if args.do_plot_occlusion_experiment:
        log('Summarizing occlusion experiments...')
        top, bottom = read_explanations(args.path_in_explanations_jsonl)
        res_top, percentages = occlusion_predictions(top)
        res_bottom, percentages = occlusion_predictions(bottom)
        f1_top = [f1_score(t[0], t[1], average='weighted') for t in res_top]
        # convert to csv
        f1_top = list(zip(percentages, f1_top))
        f1_top = [f'{tup[0]},{tup[1]}' for tup in f1_top]
        f1_top = '\n'.join(f1_top)
        f1_bottom = [f1_score(b[0], b[1], average='weighted') for b in res_bottom]
        f1_bottom = list(zip(percentages, f1_bottom))
        f1_bottom = [f'{tup[0]},{tup[1]}' for tup in f1_bottom]
        f1_bottom = '\n'.join(f1_bottom)
        with open(args.path_out_top_masked_predictions, 'w+') as fout:
            fout.write(f1_top)
            fout.close()
        with open(args.path_out_bottom_masked_predictions, 'w+') as fout:
            fout.write(f1_bottom)
            fout.close()
        # pickle.dump(f1_top, open(args.path_out_top_masked_predictions, 'w+'))
        # pickle.dump(f1_bottom, open(args.path_out_bottom_masked_predictions, 'w+'))
        if args.draw_plot:
            plt.plot(f1_top, label='top')
            plt.plot(f1_bottom, label='bottom')
            plt.legend()
            plt.show()
        log('...done summarizing occlusion experiments.')

    if args.do_convert_to_latex:
        log('Converting to latex...')
        to_latex(path_in=args.path_in_explanations_jsonl,
                 path_out=args.path_out_latex,
                 max_seq_len=args.max_seq_len,
                 crop=args.crop,
                 weight=args.weight,
                 base=args.base)
        log('...done converting to latex.')
