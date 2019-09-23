import argparse
import json
import os
import pickle
import re

import numpy as np
# noinspection PyUnresolvedReferences
import scispacy
import spacy
from gensim.models import KeyedVectors

from helpers.config import config
from helpers.log import log
from xgcn.xgraph import XNode, XGraph


def doc2graph(doc, to_lower):
    """Reads spaCy document into XGraph object."""

    def add(graph, token, to_lower=to_lower):
        id = token.i + 1  # todo: why +1? -> probably because tikz-dependency is counting from 1
        # if node already contained, something went wrong
        if graph.contains_by_id(id):
            raise AssertionError('Node contained.')
        label = token.text  # maybe do not do this here, this assumes a particular word2vec vocabulary
        if to_lower:
            label = label.lower()
        n = XNode(id=id, label=label, type='TOKEN')
        graph.add_node(n)
        return n

    graph = XGraph()
    # add all tokens
    for idx in range(len(doc)):
        token = doc[idx]
        add(graph, token)
    # add edges
    for idx in range(len(doc)):
        parent_token = doc[idx]
        parent_id = parent_token.i + 1  # todo: why +1?
        parent_node = graph.get_node(parent_id)
        for child_token in parent_token.children:
            child_id = child_token.i + 1  # todo: why +1?
            child_node = graph.get_node(child_id)
            graph.add_edge(parent_node, child_node, t=child_token.dep_)

    return graph


def line_to_graph(line, nlp, to_lower):
    """Reads PubMed text line into label, XGraph object."""
    label, sent = line.split('\t')[0], line.split('\t')[1]
    sent = sent.strip()
    doc = nlp(sent)
    g = doc2graph(doc=doc, to_lower=to_lower)
    return label, g


def preprocess_pubmed(path, to_lower, language_model):
    """Prepocesses PubMed file into list of XGraph objects."""
    pattern = "###[0-9]+$"
    pattern = re.compile(pattern)

    # retrieve output path from input path so that there is no manual mixup of the file names
    path_out = path.replace('.txt', '.p')  # iath.split('.')[:-1][0] + ".p"

    f_in = open(path, 'r')
    lines = f_in.readlines()
    graphs = []
    nlp = spacy.load(language_model, disable=['tagger',
                                              'ner',
                                              'textcat',
                                              'entity_ruler',
                                              'sentenizer',
                                              'merge_noun_chunks',
                                              'merge_entities',
                                              'merge_subtokens'])

    written = 0
    discarded = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0 or pattern.match(line.strip()):
            discarded = discarded + 1
            continue
        label, graph = line_to_graph(line.strip(), nlp, to_lower=to_lower)
        graphs.append((label, graph))
        written = written + 1
        if written % 1000 == 999:
            log('Processed {} lines'.format(written + 1))

    f_in.close()

    log("Wrote {} graphs from {} to {}, discarded {} lines.".format(written, path, path_out, discarded))

    log("Pickling to {}...".format(path_out))
    pickle.dump(graphs, open(path_out, 'wb'))
    log("...done pickling.")

    return path_out


if __name__ == '__main__':
    log('Preprocessing...')

    cfg = config('./config.json')
    print(json.dumps(cfg, indent=2))

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_wordvectors', type=bool, default=cfg['preprocessing']['word_vectors']['doit'])
    parser.add_argument('--vocab_size', type=int, default=cfg['preprocessing']['word_vectors']['vocab_size'])
    parser.add_argument('--file_word2vec', type=str, default=cfg['preprocessing']['word_vectors']['file_word2vec'])
    parser.add_argument('--dir_vocab', type=str, default=cfg['preprocessing']['word_vectors']['dir_vocab'])

    parser.add_argument('--preprocess_pubmed', type=bool, default=cfg['preprocessing']['pubmed']['doit'])
    parser.add_argument('--file_train_text', type=str, default=cfg['preprocessing']['pubmed']['file_train_text'])
    parser.add_argument('--file_dev_text', type=str, default=cfg['preprocessing']['pubmed']['file_dev_text'])
    parser.add_argument('--file_test_text', type=str, default=cfg['preprocessing']['pubmed']['file_test_text'])
    parser.add_argument('--to_lower', type=bool, default=cfg['preprocessing']['pubmed']['to_lower'])
    parser.add_argument('--language_model', type=str, default=cfg['preprocessing']['pubmed']['language_model'])

    args = parser.parse_args()

    ####################
    ### WORD VECTORS ###
    ####################
    if args.preprocess_wordvectors:
        log('Preprocessing word vectors...')

        # preprocessing step 1: build vocabulary
        VOCAB_SIZE = args.vocab_size

        fasttext = KeyedVectors.load_word2vec_format(args.file_word2vec, limit=VOCAB_SIZE)

        word2vec = {}

        # for LRP (first layer) we need a lower and an upper bound
        lower_bound = float('inf')
        upper_bound = float('-inf')

        # we sum and average all vectors here, for the unknown token
        sum_of_vectors = None
        for word in fasttext.vocab:
            word2vec[word] = np.reshape(fasttext[word], (1, -1))

            # for lrp (first layer) determine lower and upper bounds
            min_coeff = np.min(word2vec[word])
            max_coeff = np.max(word2vec[word])
            lower_bound = min_coeff if min_coeff < lower_bound else lower_bound
            upper_bound = max_coeff if max_coeff > upper_bound else upper_bound

            # sum word vectors
            if sum_of_vectors is not None:
                sum_of_vectors = sum_of_vectors + word2vec[word]
            else:
                # if this is the first word, init the sum of vectors
                sum_of_vectors = word2vec[word]

        # handle the unknown token vector
        sum_of_vectors /= VOCAB_SIZE  # normalize
        unk = '<###-unk-###>'
        word2vec[unk] = sum_of_vectors
        # note: padding token not needed, padding is performed in the course of a dataset transformation

        max_coeff = np.max(word2vec[unk])
        upper_bound = max_coeff if max_coeff > upper_bound else upper_bound
        min_coeff = np.min(word2vec[unk])
        lower_bound = lower_bound if min_coeff < lower_bound else lower_bound

        lower_bound = str(round(lower_bound, ndigits=5))
        upper_bound = str(round(upper_bound, ndigits=5))

        cfg['preprocessing']['word_vectors']['lower_bound'] = float(lower_bound)
        cfg['preprocessing']['word_vectors']['upper_bound'] = float(upper_bound)

        file_name = f'vocab_size_{VOCAB_SIZE}_min_{lower_bound}_max_{upper_bound}.p'

        path = os.path.join(args.dir_vocab, file_name)
        cfg['preprocessing']['word_vectors']['vocab'] = path

        # serialize the vocabulary and document the lower and upper bound in the name of the file
        pickle.dump(word2vec, open(path, 'wb'))

        log('Updated config.json')
        with open('config.json', 'w') as fin:
            fin.write(json.dumps(cfg, indent=2))

        log(f'Saved vocabulary of size {VOCAB_SIZE} w/ lower bound {lower_bound} and upper bound {upper_bound} to {file_name}.')
        log('...done preprocessing word vectors.')

    ##############
    ### PUBMED ###
    ##############
    if args.preprocess_pubmed:
        log('Preprocessing PubMed...')

        log(f'Preprocessing {args.file_train_text}...')
        file_train_pickle = preprocess_pubmed(path=args.file_train_text,
                                              to_lower=args.to_lower,
                                              language_model=args.language_model)
        log(f'...done preprocessing {args.file_train_text}.')

        log(f'Preprocessing {args.file_dev_text}...')
        file_dev_pickle = preprocess_pubmed(path=args.file_dev_text,
                                            to_lower=args.to_lower,
                                            language_model=args.language_model)
        log(f'...done preprocessing {args.file_dev_text}')

        log(f'Preprocessing {args.file_test_text}...')
        file_test_pickle = preprocess_pubmed(path=args.file_test_text,
                                             to_lower=args.to_lower,
                                             language_model=args.language_model)
        log(f'...done preprocessing {args.file_test_text}')

        cfg['preprocessing']['pubmed']['file_train_pickle'] = file_train_pickle
        cfg['preprocessing']['pubmed']['file_dev_pickle'] = file_dev_pickle
        cfg['preprocessing']['pubmed']['file_test_pickle'] = file_test_pickle

        log('...done preprocessing PubMed.')

        log('Updated config.')
        with open('config.json', 'w') as fin:
            fin.write(json.dumps(cfg, indent=2))
        log('...done preprocessing.')

