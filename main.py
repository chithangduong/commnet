import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable

import networkx as nx
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from tqdm import tqdm
import os

from classify import Classifier, read_node_label
from models import Model

def embed_arr_2_dict(embed_arr, G):
    embed_dict = {}
    for idx, node in enumerate(G.nodes()):
        embed_dict[str(node)] = embed_arr[idx]
    return embed_dict

def classify(vectors, args):
    if not os.path.isfile(args.classifydir +'_labels.txt'):
        return defaultdict(lambda:0)
    X, Y = read_node_label(args.classifydir +'_labels.txt')
    print("Training classifier using {:.2f}% nodes...".format(args.train_percent * 100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
    scores = clf.split_train_evaluate(X, Y, args.train_percent)
    return scores

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifydir', dest='classifydir',
            help='Directory containing classify data')

    parser.add_argument('--num-parts', dest='num_parts', type=int, default=128,
            help='Number of partitions, default=128')
    parser.add_argument('--train-perc', dest='train_percent', type=float, default=0.5,
            help='Ratio of number of labels for training, default=0.5')

    parser.add_argument('--lambda', dest='lam', type=float, default=0.7,
            help='Weight for the min-cut. 1-lam will be the weight for balance cut, default=0.7')
    parser.add_argument('--balance_node',action="store_true",
            help='Use only adj_cross')   

    parser.add_argument('--temp', dest='temp', type=float, default=10,
            help='Temperature for gumbel sinkhorn, default=10')
    parser.add_argument('-hard',action="store_true",
            help='Hard assignment of gumbel softmax') 
    parser.add_argument('--beta', type=float, default=1,
            help='Beta param of gumbel softmax, default=1')

  
    parser.add_argument('--epochs', dest='num_epochs', type=int, default=3000,
            help='Number of epochs to train, default=3000.')
    parser.add_argument('--seed', type=int, default=123,
            help='Random seed, default=123.')

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
            help='Learning rate, default=0.001.')
    parser.add_argument('--weight_decay', type=float, default=0,
            help='Weight decay, default=0.')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
            help='Gradient clipping, default=2.0.')

    return parser.parse_args()

def main(args):
    G = nx.read_edgelist(args.classifydir +'_edgelist.txt', nodetype=int)
    model = Model(nx.number_of_nodes(G), args.num_parts)
    adj = Variable(torch.FloatTensor(nx.adjacency_matrix(G).toarray()), requires_grad=False)

    if torch.cuda.is_available():
        model = model.cuda()
        adj = adj.cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.num_epochs)):
        model.zero_grad()
        
        super_adj = model(adj,temp=args.temp, hard=args.hard, beta=args.beta)
        loss = model.loss(super_adj, balance_node=args.balance_node, lam=args.lam)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if epoch %50==0:
            print("loss:", loss.item())

            vectors = embed_arr_2_dict(model.params.cpu().detach().numpy(), G)
            accs = classify(vectors, args)
            print("micro:",accs['micro'],"macro:",accs['macro'])
            
if __name__ == "__main__":

    args = arg_parse()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)

