
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--R', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--target-acc', type=float, default=70.00,
                        help='learning rate')
    parser.add_argument('--W', type=int, default=5,
                        help="number of users: ")
    parser.add_argument('--C', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--E', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--B', type=int, default=20,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.95,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--S', type=float, default=0.6,
                        help='sparcify rate(default: 0.6)')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--method', type=str, default='fed_baseline', help="name \
                                of method")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
   
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', nargs='+', default=[47], help='random seed')


    args = parser.parse_args()
    return args
