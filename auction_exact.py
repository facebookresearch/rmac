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


class Auction(Mechanism):
    def __init__(self, price_idx=0, reserve=0):
        super(Auction, self).__init__()
        self.price_idx = price_idx
        self.reserve = reserve

    def assign(self, bids):
        assert bids.ndimension() == 3
        bids += torch.rand_like(bids) * 1e-6  # break ties
        price, idx = bids.sort(1, descending=True)
        match = price[:, 0].gt(self.reserve).float()
        sale_prices = price[:, self.price_idx].clamp(self.reserve, 1e6) * match
        assignments = torch.zeros_like(bids)
        payments = torch.zeros_like(bids)
        payments.scatter_(1, idx[:, :1], sale_prices.unsqueeze(1))
        assignments = payments.ne(0).float()

        return assignments, payments

    def calc_rewards(self, assignments, payments, values):
        return (assignments * values - payments).squeeze(-1)

    def calc_statistic(self, bids, assignments, payments, values):
        # statistic is statistic
        assert payments.size(-1) == 1
        return payments.squeeze(-1).sum(-1)  # sum over (singleton) item and bidders

    def optimal_actions_uniform(self, types, bidders=2):
        if self.price_idx == 1:
            return types
        else:
            return types * (bidders - 1) / bidders


class FirstPriceAuction(Auction):
    def __init__(self, reserve=0):
        super(FirstPriceAuction, self).__init__(reserve=reserve, price_idx=0)


class SecondPriceAuction(Auction):
    def __init__(self, reserve=0):
        super(SecondPriceAuction, self).__init__(reserve=reserve, price_idx=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A3C')

    parser.add_argument('--npeople', type=int, default=1000,
                        help="Number of total bidders")
    parser.add_argument('--orig-second-price', type=int, default=0,
                        help="If 1, then the original mechanism is second-price")
    parser.add_argument('--orig-reserve', type=float, default=0,
                        help="The reserve price in the original mechanism")
    parser.add_argument('--orig-bidders', type=int, default=2,
                        help="Number of bidders in the original mechanism")
    parser.add_argument('--epsilons', default="-0.001,0,0.001",
                        help="Epsilons to evaluate for RMAC")
    parser.add_argument('--cf-reserves', default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
                        help="A list of reserves to evaluate in the counterfactual mechanism")
    parser.add_argument('--cf-bidders', default="1,2,3,4,5",
                        help="A list of number of bidders in the counterfactual mechanism")
    parser.add_argument('--beta-a', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--beta-b', type=float, default=1,
                        help="Beta a param for true type distribution")
    parser.add_argument('--uniform-types', action='store_true',
                        help="Make the true types an exact uniform distribution")
    parser.add_argument('--approx-bids', action='store_true',
                        help="If true, calculate 'observed' bids by computing an approximate Nash, rather than using the analytic solution.")
    parser.add_argument('--types-out', action='store_true',
                        help="Output true and predicted types for each person")
    parser.add_argument('--type-max', type=float, default=2,
                        help="Maximum predicted 'type' for each person.")
    parser.add_argument('--bid-max', type=float, default=1.5,
                        help="Maximum predicted bid for each person.")
    parser.add_argument('--method', default='constrain',
                        help="Adversarial method: [optimize|constrain]")
    parser.add_argument('--v', action='store_true', default=False, help="verbose")

    opt = parser.parse_args()

    true_types = torch.from_numpy(np.random.beta(opt.beta_a, opt.beta_b, size=opt.npeople)).float()
    # exact linear types
    if opt.uniform_types:
        true_types = torch.arange(0, 1, 1/opt.npeople)

    original_auction = Auction(price_idx=opt.orig_second_price, reserve=opt.orig_reserve)

    typeD = Discretization(max=opt.bid_max)
    bidD = Discretization(max=opt.bid_max)

    rmac = RMAC(typeD, bidD, method=opt.method, v=opt.v)

    assert opt.orig_second_price == 1 or (opt.orig_reserve == 0 and opt.beta_a == 1 and opt.beta_b == 1) or opt.approx_bids, "Don't know how to compute bids analytically"
    if opt.approx_bids:
        observed_bids = rmac.compute_nash(
            true_types, original_auction, opt.orig_bidders, epochs=1000)
    else:
        observed_bids = original_auction.optimal_actions_uniform(true_types)


    type_grid = typeD.grid()
    all_bids = bidD.grid()
    n_batches = type_grid.shape[0]

    # Step 1: Get reward matrix in original game
    # optimal policy is just the max over bids for each type
    orig_reward_matrix, avg_rev = rmac.get_reward_matrix(
        original_auction, opt.orig_bidders, observed_bids)

    optimal_reward, optimal_action = orig_reward_matrix.max(1)

    tag_headers = ["orig_price_idx", "orig_reserve", "cf_price_idx", "cf_reserve"]
    print(",".join(tag_headers + rmac.get_headers()))
    if opt.types_out:
        print(",".join(["types"] + tag_headers + rmac.get_types_out_headers()))
    sys.stdout.flush()

    EPSILONS = [float(x) for x in opt.epsilons.split(',')]

    for cf_reserve in [float(x) for x in opt.cf_reserves.split(',') if len(x) > 0]:
        cf_auction = SecondPriceAuction(reserve=cf_reserve)

        for eps in EPSILONS:
            rmac.calc_range(true_types, observed_bids, orig_reward_matrix,
                           cf_auction, epochs=500,
                           eps=eps, types_out=opt.types_out,
                           tag="%d,%.2g,%d,%.2g," % (original_auction.price_idx, original_auction.reserve, cf_auction.price_idx, cf_auction.reserve))

    for cf_bidders in [int(x) for x in opt.cf_bidders.split(',') if len(x) > 0]:
        cf_auction = FirstPriceAuction()

        for eps in EPSILONS:
            rmac.calc_range(true_types, observed_bids, orig_reward_matrix,
                           cf_auction, epochs=1000, cf_participants=cf_bidders,
                           eps=eps, types_out=opt.types_out,
                           tag="%d,%.2g,%d,%.2g," % (original_auction.price_idx, original_auction.reserve, cf_auction.price_idx, cf_auction.reserve))
