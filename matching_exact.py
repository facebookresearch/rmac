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


def arange(*args):
    return torch.arange(*args).long()

def order(bids):
    return bids.sort(-1, descending=True)[1]


class Matching(Mechanism):
    def __init__(self, N=3, statistic="first_choice"):
        super().__init__()
        self.statistic = statistic
        self.N = N

    def calc_rewards(self, assignments, payments, values):
        return (assignments * values).sum(-1) - payments.squeeze(-1)

    def calc_statistic(self, ranks, assignments, payments, values,
                       statistic=None):
        if statistic is None:
            statistic = self.statistic
        if statistic == "first_choice":
            return (values * assignments).sum(-1).eq(values.max(-1)[0]).float().mean(-1)
        elif statistic == "truthful":
            return values.sort(-1, descending=True)[1].eq(ranks).sum(-1).eq(3).float().mean(-1)
        elif statistic == "welfare":
            return (assignments * values).sum(-1).mean(-1)
        else:
            assert False, "Unknown statistic: %s" % statistic

    def optimal_actions_uniform(self, types, bidders=2):
        return types.sort(-1, descending=True)[1]  # idk truthful?


class BostonMatching(Matching):

    def assign(self, ranks):
        assert(ranks.ndimension() == 3)
        B = ranks.size(0)
        assignments = torch.zeros_like(ranks).float()
        agent_order = torch.rand(3, ranks.size(0)).sort(0)[1]
        for rank in range(self.N):
            for agent in agent_order:
                item = ranks[arange(B), agent, rank]
                taken = (assignments[arange(B), agent, :].sum(1) + assignments[arange(B), :, item].sum(1)).clamp_(max=1)
                assignments[arange(B), agent, item] = 1 - taken

        return assignments, assignments.sum(-1).zero_()


class RSDMatching(Matching):
    def assign(self, ranks):
        assert(ranks.ndimension() == 3)
        B = ranks.size(0)
        assignments = torch.zeros_like(ranks).float()
        agent_order = torch.rand(3, ranks.size(0)).sort(0)[1]
        for dictator in agent_order:
            for rank in range(self.N):
                item = ranks[arange(B), dictator, rank] # .unsqueeze(1)
                taken = (assignments[arange(B), dictator, :].sum(1) + assignments[arange(B), :, item].sum(1)).clamp_(max=1)
                assignments[arange(B), dictator, item] = 1 - taken
        return assignments, assignments.sum(-1).zero_()


def str2mech(name):
    if name == 'boston':
        return BostonMatching
    elif name == 'RSD':
        return RSDMatching
    else:
        assert False

class ListDiscretization(Discretization):

    def __init__(self, vals, idxvals):
        self._vals = vals
        steps = int(idxvals.max()) + 1
        self._val2idx = torch.LongTensor(steps, steps, steps).fill_(-1)
        self._val2idx[tuple(idxvals.t())] = torch.arange(idxvals.size(0)).long()

        # sanity check
        idxs = torch.arange(vals.size(0)).long()
        assert (idxs - self.val2idx(self.idx2val(idxs))).abs().max() == 0, "%s %s %s" % (
            idxs, self.idx2val(idxs), self.val2idx(self.idx2val(idxs)))

    def idx2val(self, idx):
        res = self._vals[idx.view(-1)]
        return res.view(*(list(idx.size()) + [self.Z]))

    def val2idx(self, val):
        res = self._val2idx[tuple(val.view(-1, self.Z).t())]

        assert res.min() >= 0
        assert res.max() < self._vals.size(0)
        return res.view(*(val.size()[:-1]))

    def grid(self):
        return self._vals.clone()


class Sum1Discretization(ListDiscretization):

    def __init__(self, Z=3, step=0.25):
        self.Z = Z
        self.nsteps = int(1 / step)
        self.step = step

        self.Z = Z
        assert Z == 3
        vals = []
        for i in range(self.nsteps + 1):
            for j in range(self.nsteps + 1):
                if i + j <= self.nsteps + 1e-6:
                    vals.append([i, j, self.nsteps - i - j])

        vals = torch.FloatTensor(vals) * step
        idxvals = (vals / step + 0.5).long()
        for i in range(self.Z):
            vals[:, i] += i * self.step / self.Z  # remove degeneracy
        super().__init__(vals, idxvals)

    def val2idx(self, val):
        val = val.clone()
        for i in range(self.Z):
            view = val.select(-1, i)
            view -= i * self.step / self.Z  # remove degeneracy

        val = (val / self.step + 0.5).long()
        assert val.ndimension() == 2
        val.view(-1, self.Z)[:, -1] = self.nsteps - val.view(-1, self.Z)[:, :-1].sum(1)

        return super().val2idx(val)


class PermDiscretization(ListDiscretization):

    def __init__(self, Z=3):
        assert Z == 3
        self.Z = Z
        vals = torch.LongTensor([
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
            [1, 2, 0],
            [2, 0, 1]
        ]
        )

        super().__init__(vals, vals)

class ContinuousPermDiscretization(ListDiscretization):

    def __init__(self, Z=3, step=0.1, offsets=[0, 1, 2]):
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] - offsets[i] >= 1

        self.offsets = torch.Tensor(offsets).view(1, -1)
        self.step = step
        nsteps = int(1 / step)
        self.nsteps = nsteps
        assert Z == 3
        self.Z = Z
        self.perms = torch.LongTensor([
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
            [1, 2, 0],
            [2, 0, 1]
        ])
        self._perm2idx = torch.LongTensor(self.Z, self.Z, self.Z).fill_(-1)
        self._perm2idx[tuple(self.perms.t())] = torch.arange(self.perms.size(0)).long()

        idxs = torch.stack([
            torch.Tensor([0]).unsqueeze(1).expand(nsteps, nsteps),
            torch.arange(nsteps, dtype=torch.float32).unsqueeze(1).expand(nsteps, nsteps),
            torch.arange(nsteps, dtype=torch.float32).unsqueeze(0).expand(nsteps,nsteps)], 2).view(-1, 3)
        values = idxs * step + self.offsets

        res, idxres = [], []
        for p in self.perms:
            res.append( torch.stack([values[:, p[0]], values[:, p[1]], values[:, p[2]]], 1) )
            idxres.append( torch.stack([idxs[:, p[0]], idxs[:, p[1]], idxs[:, p[2]]], 1) )
        vals = torch.cat(res, 0)

        super().__init__(vals, idxs.long())

    def val2idx(self, _val):
        val = _val.clone().view(-1, self.Z)
        sv, order = val.sort(1)
        order = order.sort(1)[1]   # LOLOL
        which_perm = self._perm2idx[tuple(order.view(-1, self.Z).t())]
        deltas = (sv - self.offsets) / self.step
        deltas = (deltas + 0.5).long()

        delta_idx = super().val2idx(deltas)
        idx = delta_idx + self.nsteps ** (self.Z - 1) * which_perm
        return idx.view_as(val.select(-1, 0))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--orig-mech', default='boston',
                        help="[boston|RSD]")
    parser.add_argument('--cf-mech', default='RSD',
                        help="[boston|RSD]")
    parser.add_argument('--step', type=float, default=1)
    parser.add_argument('--npeople', type=int, default=1000,
                        help="Number of total voters")
    parser.add_argument('--epsilons', default="-1,-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,-0.001,-0.0001,0,0.0001,0.001,0.01,0.02,0.05,0.1,0.2,0.5,1",
                        help="Epsilons to evaluate for RMAC")
    parser.add_argument('--beta-a', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--beta-b', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--types-out', action='store_true',
                        help="Output true and predicted types for each person.")
    parser.add_argument('--method', default='constrain',
                        help="Adversarial method: [optimize|constrain]")
    parser.add_argument('--offsets', default=[0, 1, 2], type=float, nargs=3)
    parser.add_argument('--nepochs', type=int, default=1000)
    parser.add_argument('--statistic', default="first_choice",
                        help="first_choice|truthful|welfare")
    parser.add_argument('--reward-samples', type=int, default=10)
    parser.add_argument('--v', action='store_true', default=False, help="verbose")


    opt = parser.parse_args()
    print("# %s" % opt)

    offsets = opt.offsets
    assert offsets[0] == 0

    true_types = torch.from_numpy(np.random.beta(opt.beta_a, opt.beta_b, size=opt.npeople * 3)).float().view(-1, 3)
    true_types[:, 0] = 0
    true_types *= (1 - opt.step)
    true_types[:, 1] += offsets[1]
    true_types[:, 2] += offsets[2]

    original_mech = str2mech(opt.orig_mech)(statistic=opt.statistic)

    typeD = ContinuousPermDiscretization(step=opt.step, offsets=offsets) # Sum1Discretization(Z=3, step=0.1)
    actD = PermDiscretization(Z=3)
    rmac = RMAC(typeD, actD, method=opt.method)

    observed_bids = rmac.compute_nash(
        true_types, original_mech, 3, epochs=opt.nepochs)

    type_grid = typeD.grid()
    all_bids = actD.grid()
    n_batches = type_grid.shape[0]

    # Step 1: Get reward matrix in original game
    # optimal policy is just the max over bids for each type

    orig_reward_matrix, orig_stat_matrix = 0, 0
    for i in range(opt.reward_samples):
        # It's cheating to get the stat matrix using the true types.
        # but we don't need types for this statistic so just use zeros
        dummy_types = torch.zeros_like(true_types)
        orig_reward_matrix_sample, orig_stat_matrix_sample = rmac.get_reward_matrix(
            original_mech, 3, observed_bids, type_distribution=dummy_types)
        orig_reward_matrix += orig_reward_matrix_sample / opt.reward_samples
        orig_stat_matrix += orig_stat_matrix_sample / opt.reward_samples

    print("True orig statistic: %g , truthful: %g" % (
        rmac.get_statistic(original_mech, observed_bids, true_types, 3),
        rmac.get_statistic(original_mech, observed_bids, true_types, 3, statistic='truthful')))

    tag_headers = ["orig_mech", "cf_mech"]
    print(",".join(tag_headers + rmac.get_headers()))
    sys.stdout.flush()

    EPSILONS = [float(x) for x in opt.epsilons.split(',')]

    cf_mech = str2mech(opt.cf_mech)(statistic=opt.statistic)

    if opt.v:
        print('type distribution:')
        print(typeD.grid())
        print('action distribution:')
        print(actD.grid())
        print("reward matrix")
        print(orig_reward_matrix)
        print("stat matrix")
        print(orig_stat_matrix)
        print("true types | observed bids")
        print(count_unique_rows(torch.cat((true_types.float(), observed_bids.float()), dim=1)))

    for eps in EPSILONS:
        tag = opt.orig_mech + ',' + opt.cf_mech + ','
        guessed_types, guessed_bids, est_statistic = rmac.calc_range(
            true_types, observed_bids, orig_reward_matrix,
            cf_mech, epochs=opt.nepochs, eps=eps, types_out=opt.types_out,
            cf_participants=3,
            orig_participants=3,
            orig_mech=original_mech,
            orig_statistic=-orig_stat_matrix, # use negative because we want to compute cf_stat-orig_stat
            tag=tag)

        orig_statistic = rmac.get_statistic(original_mech, observed_bids, guessed_types, 3)
        orig_truthful = rmac.get_statistic(original_mech, observed_bids, guessed_types, 3, statistic='truthful')
        est_truthful = rmac.get_statistic(cf_mech, guessed_bids, guessed_types, 3, statistic='truthful')

        if opt.v:
            print("eps= %g ; truthful %g -> %g ; statistic %g -> %g" % (
                eps, orig_truthful, est_truthful, orig_statistic, est_statistic))

            print("observed bid| guessed type| guessed bid | count")
            print("------------+-------------+-------------+------")
            unique_ppl = count_unique_rows(torch.cat((observed_bids.float(),
                                                      guessed_types.float(),
                                                      guessed_bids.float()),
                                                     dim=1))
            for row in unique_ppl:
                print("%3d %3d %3d | %3d %3d %3d | %3d %3d %3d | %4d" % tuple(row))
            print("")
