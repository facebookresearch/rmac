#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

F=matching_results
for STAT in truthful welfare; do
    python matching_exact.py --orig-mech boston --cf-mech rsd > $F/boston_to_rsd_${STAT}.log
    python matching_exact.py --orig-mech rsd --cf-mech boston > $F/rsd_to_boston_${STAT}.log
done
