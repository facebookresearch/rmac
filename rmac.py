#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import sys

def sgn(x):
    return x / (abs(x) + 1e-20)


def count_unique_rows(T):
    # newer versions of pytorch actually support this as
    # T.unique(dim=1, return_count=True) but I don't have that new version
    assert T.ndimension() == 2
    unique_rows = {}
    for row in T:
        rowl = tuple(row.tolist())
        unique_rows.setdefault(rowl, 0)
        unique_rows[rowl] += 1

    res = torch.Tensor([list(k) + [v] for k, v in unique_rows.items()])
    # now sort by count, descending
    order = res[:, -1].sort(0, descending=True)[1]
    return res[order]


class Mechanism(nn.Module):

    def assign(self, actions):
        raise NotImplementedError()

    def calc_rewards(self, assignments, payments, values):
        raise NotImplementedError()

    def calc_statistic(self, actions, assignments, payments, values):
        raise NotImplementedError()


def plot():
    plt.legend()
    plt.draw()
    plt.pause(0.01)
    plt.clf()
    plt.show()


class Discretization(object):
    def __init__(self, step=0.01, min=0, max=2):
        self.step = step
        self.min = min
        self.max = max

    def val2idx(self, val):
        assert self.min == 0
        return ((val - self.min) / self.step + 0.5).long() #.clamp(0, self.N - 1)

    def idx2val(self, idx):
        return (idx.float() * self.step) + self.min

    def grid(self):
        return torch.arange(self.min, self.max + self.step, self.step)

    def size(self):
        return len(self.grid())


class RMAC(object):

    def __init__(self, typeD, actD, v=False, method=None):
        self.typeD = typeD
        self.type_grid = typeD.grid()
        self.actD = actD
        self.action_grid = actD.grid()
        self.v = v
        self.method = method

    def get_reward_matrix(self, mech, participants, action_distribution, type_distribution=None):
        NB = len(self.action_grid)
        NT = len(self.type_grid)
        NO = len(action_distribution)
        NE = self.action_grid.nelement() // NB
        assert self.type_grid.nelement() / NT == NE
        assert action_distribution.nelement() / NO == NE

        actions = self.action_grid.view(NB, 1, 1, NE).expand(NB, NO, 1, NE)
        other_perm = [torch.randperm(NO) for _ in range(participants - 1)]
        for i in range(participants - 1):
            actions_other = action_distribution[other_perm[i]].view(1, NO, 1, NE).expand(NB, NO, 1, NE)
            actions = torch.cat((actions, actions_other), dim=2)
        actions = actions.unsqueeze(0).contiguous()
        # actions: num_actions * num_people x participants

        assignments, payments = mech.assign(actions.view(-1, participants, NE))
        assignments = assignments.view(1, NB, NO, participants, NE)
        payments = payments.view(1, NB, NO, participants, 1)
        assignments0 = assignments[:, :, :, 0] # 1 x NB x NO x NE
        payments0 = payments[:, :, :, 0] # 1 x NB x NO x 1
        values_v = self.type_grid.view(NT, 1, 1, NE)

        rewards0 = mech.calc_rewards(assignments0, payments0, values_v)
        reward_matrix = rewards0.mean(2)

        if type_distribution is not None:
            types = self.type_grid.view(NT, 1, 1, NE).expand(NT, NO, 1, NE)
            for i in range(participants - 1):
                types_other = type_distribution[other_perm[i]].view(1, NO, 1, NE).expand(NT, NO, 1, NE)
                types = torch.cat((types, types_other), dim=2)
            types = types.contiguous().view(NT, 1, NO, participants, NE)
            statistic = mech.calc_statistic(actions, assignments, payments, types)
            assert statistic.ndimension() == 3
            statistic = statistic.expand(NT, NB, NO)

            avg_statistic = statistic.mean(2)
            # print(avg_statistic)
        else:
            avg_statistic = None
        return reward_matrix, avg_statistic

    def get_regret_matrix(self, reward_matrix):
        assert reward_matrix.ndimension() == 2
        optimal_reward = reward_matrix.max(1)[0].unsqueeze(1)
        return optimal_reward - reward_matrix

    def adversarialize_regret(self, regret, eps, stat, dim0=False):
        if eps == 0 or type(stat) == int and stat == 0:
            return regret
        if self.method == 'constrain':
            if dim0:
                regret = regret - regret.min(0)[0].unsqueeze(0)
            invalid_types_given_action = regret.gt(abs(eps))
            adv_regret = -sgn(eps) * stat + (stat.max() - stat.min() + 1e3) * invalid_types_given_action.float()
        elif self.method == 'optimize':
            adv_regret = regret - (eps * stat).unsqueeze(1)
        else:
            assert False, self.method
        return adv_regret

    def compute_types_from_actions(self, observed_actions, reward_matrix, eps=0, stat=None):
        observed_action = self.actD.val2idx(observed_actions)

        regret = self.get_regret_matrix(reward_matrix)

        adv_regret = self.adversarialize_regret(regret, eps, stat, dim0=True)
        guessed_regret = adv_regret[:, observed_action]

        guessed_regret += torch.rand(guessed_regret.size()) * 1e-7

        best_observed_type_idx = guessed_regret.min(0)[1]
        best_observed_type = self.typeD.idx2val(best_observed_type_idx)
        return best_observed_type

    def _concat(self, L, e):
        return torch.cat([L, e.unsqueeze(1)], 1) if L is not None else e.clone().unsqueeze(1)

    # Updating CF strategy
    def compute_adversarial_nash(self,
                                 guessed_types,
                                 cf_mech,
                                 cf_participants,
                                 epochs=500,
                                 eps=0,
                                 observed_actions=None,
                                 orig_reward_matrix=None,
                                 orig_statistic=None):

        cf_reward_matrix = torch.rand(self.type_grid.size(0), self.action_grid.size(0))
        cf_stat_matrix = 0
        fptypes, fpactions = None, None
        c = 1
        grange = torch.arange(guessed_types.size(0)).long()
        for i in range(epochs):
            if self.v and i % 10 == 0 and i > 0:
                log_stat = self.get_statistic(cf_mech,
                                              fpactions[grange, rand_hist],
                                              fptypes[grange, rand_hist],
                                              participants=cf_participants,
                                              num_samples=50000)
                print("Epoch %d/%d : %g" % (i, epochs, log_stat))

            fptypes = self._concat(fptypes, guessed_types)

            guessed_type_idx = self.typeD.val2idx(guessed_types)

            cf_regret_matrix = self.get_regret_matrix(cf_reward_matrix)

            cf_adv_regret = self.adversarialize_regret(cf_regret_matrix,
                                                       eps,
                                                       cf_stat_matrix)

            cf_optimal_action_map = cf_adv_regret.min(1)[1]

            guessed_regret_matrix = cf_adv_regret[guessed_type_idx]

            cf_regrets, cf_action_distribution = (guessed_regret_matrix +
                                                  torch.rand(guessed_regret_matrix.size()) * 1e-7).min(1)

            cf_action_distribution = self.actD.idx2val(cf_action_distribution)

            fpactions = self._concat(fpactions, cf_action_distribution)

            # compute the reward matrix R(type, action) given the current types
            rand_hist = torch.rand(guessed_types.size(0)).mul(i).long()
            cf_reward_matrix, cf_stat_matrix = self.get_reward_matrix(
                cf_mech, cf_participants,
                fpactions[grange, rand_hist],
                type_distribution=fptypes[grange, rand_hist])

            c = c + 1

            if eps != 0:

                avg_stat_by_type = cf_stat_matrix[torch.arange(len(self.type_grid)).long(), cf_optimal_action_map]
                stat_matrix = avg_stat_by_type.unsqueeze(1).expand_as(orig_reward_matrix).contiguous()

                if orig_statistic is not None:
                    stat_matrix = stat_matrix + orig_statistic

                guessed_types = self.compute_types_from_actions(observed_actions,
                                                                orig_reward_matrix,
                                                                eps=eps,
                                                                stat=stat_matrix)

        # pick only from the second half of the history!
        rand_recent_hist = torch.rand(guessed_types.size(0)).mul(i/2).add(i/2).long()
        recent_hist_types = fptypes[grange, rand_recent_hist]
        recent_hist_actions = fpactions[grange, rand_recent_hist]

        return recent_hist_types, recent_hist_actions

    def compute_nash(self, types, mech, participants, epochs):
        orig_types, equilibrium_actions = self.compute_adversarial_nash(types, mech, participants, epochs=epochs)
        assert orig_types.eq(types).all()
        return equilibrium_actions

    def get_statistic(self, mech, actions, types, participants, num_samples=50000, **kwargs):
        to_sample = np.random.choice(actions.shape[0],
                                     num_samples * participants,
                                     replace=True).reshape(num_samples, participants)

        sampled_actions = actions[to_sample]
        sampled_types = types[to_sample]
        if sampled_actions.ndimension() == 2:
            sampled_actions = sampled_actions.unsqueeze(-1)
        if sampled_types.ndimension() == 2:
            sampled_types = sampled_types.unsqueeze(-1)

        assignments, payments = mech.assign(sampled_actions)
        statistic = mech.calc_statistic(sampled_actions, assignments, payments, sampled_types, **kwargs)
        assert statistic.ndimension() == 1
        return statistic.mean()

    def make_plots(self, true_types, guessed_types, observed_actions, guessed_actions, tag, eps):

        plt.figure()
        plt.scatter(true_types.detach().numpy(), guessed_types.detach().numpy(), label='true_t:t')
        plt.xlabel('true type')
        plt.ylabel('guessed type')
        plt.savefig('types_%seps%g.png' % (tag, eps))
        plt.clf()

        plt.figure()
        plt.scatter(true_types.detach().numpy(), observed_actions.detach().numpy(), label='t:orig_actions')
        plt.xlabel('true type')
        plt.ylabel('observed action')
        plt.savefig('orig_actions_%seps%g.png' % (tag, eps))
        plt.clf()

        plt.figure()
        plt.scatter(true_types.detach().numpy(), guessed_actions.detach().numpy(), label='t:cf_action')
        plt.xlabel('true type')
        plt.ylabel('counterfactual action')
        plt.savefig('cf_actions_%seps%g.png' % (tag, eps))
        plt.clf()


    def get_headers(self):
        return ["cf_participants","eps","true_stat","est_stat","XXX","type_regret_mean","type_regret_max","policy_regret_mean","policy_regret_max"]

    def get_types_out_headers(self):
        return ["cf_participants","eps","true_type","observed_bid","guessed_type","cf_bid"]

    def calc_range(self,
                   true_types,
                   observed_actions,
                   orig_reward_matrix,
                   cf_mech,
                   epochs,
                   eps=0,
                   cf_participants=2,
                   orig_mech=None,
                   orig_statistic=None,
                   orig_participants=None,
                   types_out=None,
                   plot=False,
                   tag=''):

        guessed_types = self.compute_types_from_actions(observed_actions, orig_reward_matrix)
        assert guessed_types.ndimension() == true_types.ndimension()

        guessed_types, guessed_actions = self.compute_adversarial_nash(
            guessed_types,
            cf_mech, cf_participants,
            eps=eps,
            epochs=epochs,
            observed_actions=observed_actions,
            orig_reward_matrix=orig_reward_matrix,
            orig_statistic=orig_statistic)

        assert guessed_types.ndimension() == true_types.ndimension()

        cf_reward_matrix, cf_stat_matrix = self.get_reward_matrix(
            cf_mech, cf_participants,
            guessed_actions,
            type_distribution=guessed_types)
        cf_regret_matrix = self.get_regret_matrix(cf_reward_matrix)
        cf_adv_regret = self.adversarialize_regret(cf_regret_matrix,
                                                   eps,
                                                   cf_stat_matrix)
        guessed_type_idx = self.typeD.val2idx(guessed_types)
        guessed_adv_regret = cf_adv_regret[guessed_type_idx]

        if plot:
            self.make_plots(true_types, guessed_types, observed_actions, guessed_actions, tag, eps)

        # print some statistics
        cf_optimal_actions = cf_mech.optimal_actions_uniform(true_types, cf_participants)
        true_statistic = self.get_statistic(cf_mech, cf_optimal_actions, true_types, cf_participants)
        est_statistic = self.get_statistic(cf_mech, guessed_actions, guessed_types, cf_participants)

        if orig_statistic is not None and orig_mech is not None:
            # Note; this is not the same as
            # orig_statistic[guessed_types, guessed_actions]
            # because the orig_statistic matrix assumes that all the *other*
            # participants have their original guessed type. As long as the form
            # is separable, it will still be the same up to a constant, but won't
            # give the right answer for logging
            log_stat = self.get_statistic(orig_mech,
                                          observed_actions,
                                          guessed_types,
                                          orig_participants)
            opt_old_statistic = "%.2g," % log_stat
        else:
            opt_old_statistic = ""

        original_regrets = self.get_regret_matrix(orig_reward_matrix)
        cf_regrets_exact = self.get_regret_matrix(cf_reward_matrix)
        observed_actions = self.actD.val2idx(observed_actions)
        regrets_per_action = original_regrets[self.typeD.val2idx(guessed_types)]
        observed_type_regrets = regrets_per_action.gather(1, observed_actions.unsqueeze(1))

        observed_policy_regrets = cf_regrets_exact[guessed_type_idx].gather(
            1, guessed_adv_regret.min(1)[1].unsqueeze(1))

        print("%s%d,%.3g,%.4g,%s%.4g,XXX,%.4g,%.4g,%.4g,%.4g"% (
            tag, cf_participants, eps, true_statistic, opt_old_statistic, est_statistic,
            observed_type_regrets.mean(), observed_type_regrets.max(),
            observed_policy_regrets.mean(), observed_policy_regrets.max()))
        sys.stdout.flush()

        if types_out:
            for t, o, g, c in zip(true_types, observed_actions, guessed_types, guessed_actions):
                print("types,%s%d,%.4g,%.4g,%.4g,%.4g,%.4g" %
                      (tag, cf_participants, eps, t, o, g, c))

        return guessed_types, guessed_actions, est_statistic
