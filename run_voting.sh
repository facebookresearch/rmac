#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

F=voting_results
mkdir -p $F
for O in mean median vcg; do
  echo $O;
  python voting_exact.py --orig-mech $O --cf-mech median --orig-voters 11 --cf-voters 1 --method constrain --epsilons "0.0001,0,-0.0001" --types-out > $F/voting_${O}.log
done
