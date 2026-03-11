#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2024 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import torch
import argparse

join=os.path.join

if 'save_weights_torch' not in locals() and 'save_weights_torch' not in globals():
    from dafne_models.common import save_weights_torch

from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_ID", type=str, required=True, help="ID model string")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to the model file to use")
    parser.add_argument("--save_weights_path", "-s", type=str, required=True, help="Path to the save weights")
    args = parser.parse_args()

    save_weights_torch(model_id=args.model_ID,
                    model_path=args.model_path,
                    final_weights_path=args.save_weights_path,
                    model_type=DynamicEnsembleModel
                    )
