import _pickle as pickle
import argparse
import json
import pickle
import warnings

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from helpers.config import config
from helpers.log import log
from xgcn.xgcn import XGCN
from xgcn.xgraph import XSample, Pad, ToTensor, LabelToOneHot


class PubMedDataset(Dataset):
    """Converts Pubmed data into tesnors."""

    def __init__(self,
                 path_pickle,
                 path_word2vec,
                 pad=150,
                 crop=-1):
        self.path_word2vec = path_word2vec
        self.label2vec = pickle.load(open(path_word2vec, 'rb'))
        self.path_pickle = path_pickle
        self.label_graph_tuples = pickle.load(open(self.path_pickle, 'rb'))
        self.label2onehot = LabelToOneHot(classes=PubMedDataset.classes())
        self.crop = crop
        self.pad = pad
        self.ops = [Pad(self.pad), ToTensor(), self.label2onehot]
        self.transforms = transforms.Compose(self.ops)

    @staticmethod
    def classes():
        return ["METHODS", "RESULTS", "CONCLUSIONS", "BACKGROUND", "OBJECTIVE"]

    def __len__(self):
        if self.crop > 0:
            return self.crop
        return len(self.label_graph_tuples)

    def __getitem__(self, index):
        label, graph = self.label_graph_tuples[index]
        embedding = graph.E(label2vec=self.label2vec)
        adjacency = graph.A_tilde()
        assert (embedding.shape[0] == adjacency.shape[0])
        xsample = XSample(embedding, adjacency, label)
        xsample = self.transforms(xsample)
        return xsample.EMBEDDING, xsample.ADJACENCY, xsample.LABEL


def loader(path_train_file: str,
           path_dev_file: str,
           path_test_file: str,
           path_word2vec: str,
           pad: int,
           crop_train: int,
           crop_dev: int,
           crop_test: int,
           batch_size: int,
           num_workers: int):
    """Returns train, dev and test set pubmed data laoders."""

    pin_memory = torch.cuda.is_available()

    dataset_train_pubmed = PubMedDataset(path_pickle=path_train_file,
                                         path_word2vec=path_word2vec,
                                         pad=pad,
                                         crop=crop_train)
    dataloader_train_pubmed = DataLoader(dataset_train_pubmed,
                                         batch_size=batch_size,
                                         pin_memory=pin_memory,
                                         num_workers=num_workers)

    dataset_dev_pubmed = PubMedDataset(path_pickle=path_dev_file,
                                       path_word2vec=path_word2vec,
                                       pad=pad,
                                       crop=crop_dev)
    dataloader_dev_pubmed = DataLoader(dataset_dev_pubmed,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers)

    dataset_test_pubmed = PubMedDataset(path_pickle=path_test_file,
                                        path_word2vec=path_word2vec,
                                        pad=pad,
                                        crop=crop_test)
    dataloader_test_pubmed = DataLoader(dataset_test_pubmed,
                                        batch_size=batch_size,
                                        pin_memory=pin_memory,
                                        num_workers=num_workers)

    return dataloader_train_pubmed, dataloader_dev_pubmed, dataloader_test_pubmed


def validate(xgcn, dataloader, device):
    """Determines performance of xgcn model on the data provided by dataloader."""

    log('Validating...')
    xgcn.eval()
    xgcn.to(device)
    outputs = None
    targets = None
    for (embeddings, adjacencies, labels) in tqdm(dataloader):
        embeddings = embeddings.to(device)
        adjacencies = adjacencies.to(device)
        labels = labels.to(device)
        if targets is None:
            targets = labels
        else:
            targets = torch.cat((targets, labels))

        output = xgcn(embeddings, adjacencies)
        output = torch.argmax(output, dim=1)

        if outputs is None:
            outputs = output
        else:
            outputs = torch.cat((outputs, output))

    outputs = outputs.tolist()
    targets = targets.tolist()

    outputs, targets = zip(*((output, target) for output, target in zip(outputs, targets))) # todo what does this do?
    outputs = list(outputs)
    targets = list(targets)

    f_score_micro = f1_score(y_pred=outputs, y_true=targets, average='micro')
    f_score_macro = f1_score(y_pred=outputs, y_true=targets, average='macro')
    f_score_weighted = f1_score(y_pred=outputs, y_true=targets, average='weighted')
    log('...done validating.')

    return {'micro': f_score_micro,
            'macro': f_score_macro,
            'weighted': f_score_weighted}


def report(epoch, split, scores):
    log("Epoch: {} Split: {} F-micro: {:.3f} F-macro: {:.3f} F-weighted: {:.3f}"
        .format(epoch, split, scores['micro'], scores['macro'], scores['weighted']))


def train(loader_train,
          loader_dev,
          path_model,
          epochs,
          batch_size,
          pad,
          nfeat,
          nhid,
          patience,
          metric,
          random_seed):
    """Trains an GCN."""

    nclasses = 5
    assert metric in ["weighted", "macro", "micro"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        log("Set random cuda seed to {}".format(random_seed))
    else:
        torch.manual_seed(random_seed)
        log("Set manual seed to {}".format(random_seed))

    log("Training on device {}".format(device))

    xgcn = XGCN(nfeat=nfeat, nhid=nhid, nclass=nclasses, pad=pad, bias=None)
    xgcn.to(device)
    print(xgcn)
    optimizer = Adam(params=xgcn.parameters())  # todo pass as argument

    scores = validate(xgcn=xgcn, dataloader=loader_dev, device=device)
    report(epoch=0, split="Dev", scores=scores)

    torch.save(xgcn.state_dict(), path_model)
    log("Saved initial model to {}.".format(path_model))

    wait = 0
    score_last = float('-inf')

    running_loss = 0.0
    for epoch in range(epochs):
        xgcn.train()
        for batch_idx, (embeddings, adjacencies, labels) in enumerate(loader_train):

            embeddings = embeddings.to(device)
            adjacencies = adjacencies.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = xgcn(embeddings, adjacencies)
            loss = F.nll_loss(preds, labels)
            loss.backward()
            optimizer.step()
            xgcn.xfc.weight.data.clamp_(0)

            # print statistics
            running_loss += loss.item()
            if batch_idx % 10 == 9:
                log('[%d, %5d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, (batch_idx + 1) * batch_size, running_loss / 10))
                running_loss = 0.0

        scores = validate(xgcn=xgcn, dataloader=loader_dev, device=device)
        report(epoch=epoch + 1, split="Dev", scores=scores)

        score_current = scores[metric]

        if score_current > score_last:
            torch.save(xgcn.state_dict(), path_model)
            log("{} score improved from {:.3f} to {:.3f}. Saved model to {}."
                .format(metric, score_last, score_current, path_model))
            score_last = score_current
            wait = 0
        else:
            wait = wait + 1
            if wait >= patience:
                log("Terminating training after {} epochs w/o improvement.".format(wait))
                return xgcn

if __name__ == "__main__":
    log('Training...')
    cfg = config('./config.json')
    print(json.dumps(cfg, indent=2))

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_train_pickle', type=str, default=cfg['preprocessing']['pubmed']['file_train_pickle'])
    parser.add_argument('--file_dev_pickle', type=str, default=cfg['preprocessing']['pubmed']['file_dev_pickle'])
    parser.add_argument('--file_test_pickle', type=str, default=cfg['preprocessing']['pubmed']['file_test_pickle'])
    parser.add_argument('--vocab', type=str, default=cfg['preprocessing']['word_vectors']['vocab'])
    parser.add_argument('--pad', type=int, default=cfg['training']['pad'])
    parser.add_argument('--crop_train', type=int, default=cfg['training']['crop_train'])
    parser.add_argument('--crop_dev', type=int, default=cfg['training']['crop_dev'])
    parser.add_argument('--crop_test', type=int, default=cfg['training']['crop_test'])
    parser.add_argument('--batch_size', type=int, default=cfg['training']['batch_size'])
    parser.add_argument('--num_workers', type=int, default=cfg['training']['num_workers'])
    parser.add_argument('--path_model', type=str, default=cfg['training']['path_model'])
    parser.add_argument('--epochs', type=int, default=cfg['training']['epochs'])
    parser.add_argument('--nfeat', type=int, default=cfg['training']['nfeat'])
    parser.add_argument('--nhid', type=int, default=cfg['training']['nhid'])
    parser.add_argument('--patience', type=int, default=cfg['training']['patience'])
    parser.add_argument('--metric', type=str, default=cfg['training']['metric'])
    parser.add_argument('--random_seed', type=int, default=cfg['training']['random_seed'])
    args = parser.parse_args()

    if args.crop_test >= 0:
        warnings.warn("Cropped test set.")

    train_loader, dev_loader, test_loader = loader(path_train_file=args.file_train_pickle,
                                                   path_dev_file=args.file_dev_pickle,
                                                   path_test_file=args.file_test_pickle,
                                                   path_word2vec=args.vocab,
                                                   pad=args.pad,
                                                   crop_train=args.crop_train,
                                                   crop_dev=args.crop_dev,
                                                   crop_test=args.crop_test,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    xgcn = train(loader_train=train_loader,
                 loader_dev=dev_loader,
                 path_model=args.path_model,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 pad=args.pad,
                 nfeat=args.nfeat,
                 nhid=args.nhid,
                 patience=args.patience,
                 metric=args.metric,
                 random_seed=args.random_seed)
    log('... done training.')

    log('Validating test set...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = validate(xgcn=xgcn, dataloader=test_loader, device=device)
    report(epoch="Test", scores=scores, split='Test')
    log('...done validating test set.')

