#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import numpy as np
import argparse
import sys

from rmac import *

class Voting(Mechanism):
    def calc_rewards(self, winners, payments, values):
        diff = winners - values
        return (-diff * diff - payments).squeeze(-1)

    def calc_statistic(self, bids, winners, payments, values):
        # statistic is the voted-upon value
        assert winners.size(-1) == 1
        return winners.squeeze(-1).mean(-1)  # sum over (singleton) item and voters

class MeanVoting(Voting):
    def assign(self, votes):
        assert votes.ndimension() == 3
        return votes.mean(1, keepdim=True).expand_as(votes), torch.zeros_like(votes)

    def optimal_actions_uniform(self, types, voters=2):
        return types.clone().zero_()  # idk!

class MedianVoting(Voting):
    def assign(self, votes):
        assert votes.ndimension() == 3
        return votes.median(1, keepdim=True)[0].expand_as(votes), torch.zeros_like(votes)

    def optimal_actions_uniform(self, types, voters=2):
        return types # turthful

class VCGVoting(Voting):
    def assign(self, votes):
        assert votes.ndimension() == 3
        B = votes.size(1)
        mean = votes.mean(1, keepdim=True)
        mean_without_me = (mean * B - votes) / (B - 1 + 1e-8)

        external_cost = (votes - mean).pow(2).sum(1, keepdim=True) - (votes - mean)**2
        external_cost_without_me = torch.stack([(votes - mean_without_me[:, i:i+1]).pow(2).sum(1) -
                                       (votes[:, i]-mean_without_me[:, i])**2 for i in range(B)], dim=1)
        externality = external_cost - external_cost_without_me
        return mean.expand_as(votes), externality

    def optimal_actions_uniform(self, types, voters=2):
        return types # truthful


def str2mech(name):
    if name == 'mean':
        return MeanVoting
    elif name == 'median':
        return MedianVoting
    elif name == 'vcg':
        return VCGVoting
    else:
        assert False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--orig-mech', default='mean',
                        help="Original mechanism [mean|median|vcg]")
    parser.add_argument('--cf-mech', default='median',
                        help="Original mechanism [mean|median|vcg]")
    parser.add_argument('--orig-voters', type=int, default=3,
                        help="Number of voters in original mechanism")
    parser.add_argument('--cf-voters', type=int, default=3,
                        help="Number of voters in counterfactual mechanism")
    parser.add_argument('--npeople', type=int, default=1000,
                        help="Number of total voters")
    parser.add_argument('--epsilons', default="-1,-0.1,-0.01,0,+0.01,+0.1,+1",
                        help="Epsilons to evaluate for RMAC")
    #parser.add_argument('--cf-voters', default="1,3,5,7")
    parser.add_argument('--beta-a', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--beta-b', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--types-out', action='store_true',
                        help="Output true and predicted types for each person")
    parser.add_argument('--method', default='constrain',
                        help="Adversarial method: [optimize|constrain]")
    parser.add_argument('--hypothesis-max', type=float, default=1.5,
                        help="Maximum vote in counterfactual mechanism.")

    opt = parser.parse_args()

    true_types = torch.from_numpy(np.random.beta(opt.beta_a, opt.beta_b, size=opt.npeople)).float()

    original_mech = str2mech(opt.orig_mech)()

    typeD = Discretization(min=0, max=opt.hypothesis_max)
    actD = Discretization(min=0, max=opt.hypothesis_max)

    rmac = RMAC(typeD, actD, method=opt.method)

    observed_bids = rmac.compute_nash(
        true_types, original_mech, opt.orig_voters, epochs=10)

    type_grid = typeD.grid()
    all_bids = actD.grid()
    n_batches = type_grid.shape[0]

    # Step 1: Get reward matrix in original game
    # optimal policy is just the max over bids for each type
    orig_reward_matrix, avg_rev = rmac.get_reward_matrix(
        original_mech, opt.orig_voters, observed_bids)

    print("# %s" % opt)
    tag_headers = ["orig_mech"]
    print(",".join(tag_headers + rmac.get_headers()))
    if opt.types_out:
        print(",".join(["types"] + tag_headers + rmac.get_types_out_headers()))
    sys.stdout.flush()

    EPSILONS = [float(x) for x in opt.epsilons.split(',')]

    cf_mech = str2mech(opt.cf_mech)()

    for eps in EPSILONS:
        rmac.calc_range(true_types, observed_bids, orig_reward_matrix,
                       cf_mech, epochs=100, eps=eps, types_out=opt.types_out,
                       cf_participants=opt.cf_voters,
                       tag=opt.orig_mech + ',')
