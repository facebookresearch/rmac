#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

data = []
reserves = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0)  #FIXME
bidders = (1,2,3,4,5)  #FIXME

for i in range(1, 41):
    d = torch.load('fp%d.pt' % i)
    (hest, hex), _, reserve_points, bidder_points = d
    data.append(d)
    histx = torch.linspace(-2, 4, 100)
    with open('hist.csv', 'a') as f:
        if i == 1:
            f.write('replicate,value,exact,estimated\n')
        for a, b, c in zip(histx, hex, hest):
            f.write('%d,%g,%g,%g\n' % (i, a, b, c))
    with open('revenue_reserve.csv', 'a') as f:
        if i == 1:
            f.write('replicate,reserve,exact,estimated,est_policy,est_types\n')
        for r,(a,b,c,d) in zip(reserves, reserve_points):
            f.write("%d,%g,%g,%g,%g,%g\n" % (i, r, a, b, c, d))

    with open('revenue_bidders.csv', 'a') as f:
        if i == 1:
            f.write('replicate,num_bidders,exact,estimated,est_policy,est_types\n')
        for r,(a,b,c,d) in zip(bidders, bidder_points):
            f.write("%d,%g,%g,%g,%g,%g\n" % (i, r, a, b, c, d))

with open('types.csv', 'a') as f:
    estimated_types, true_types = zip(*(d[1] for d in data))
    for i in range(len(data[0][1][0])):
        f.write("%g,%s\n" % (data[0][1][1][i], ",".join([str(float(data[j][1][0][i])) for j in range(len(data))])))


