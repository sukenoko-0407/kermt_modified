# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import shutil
from functools import partial
import os
import numpy as np
import torch
from rdkit import RDLogger
import json

from kermt.util.parsing import parse_args, get_newest_train_args
from kermt.util.utils import create_logger
from kermt.util.hpo_space import resolve_hpo_space, suggest_from_space
from task.cross_validate import cross_validate
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction
from kermt.data.torchvocab import MolVocab
from task.train import run_training

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)

def objective_all(trial, args, logger, hpo_space, hpo_space_source):

    ## Setup optuna stuff
    # Change save_dir to temp location
    print(f"Current trial.number: {trial.number}")
    failed_trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
    print(f"failed_trial_number: {failed_trial_number}")

    trial_number = trial.number
    parent_save_dir = args.save_dir
    args.save_dir = os.path.join(args.save_dir, f"tmp_trial_{trial_number}")
    print(f"Saving temporarily to {args.save_dir}")

    ## HPO to tune
    sampled_params = suggest_from_space(trial=trial, space=hpo_space)
    for name, value in sampled_params.items():
        setattr(args, name, value)

    max_lr = sampled_params["max_lr"]
    final_lr_factor = sampled_params["final_lr_factor"]
    init_lr_factor = hpo_space["init_lr_factor"]

    args.final_lr_factor = final_lr_factor
    args.max_lr = max_lr
    args.init_lr = max_lr / init_lr_factor
    args.final_lr = max_lr / final_lr_factor

    print("Current set of hyperparameters used:")
    print(args)
    ensemble_scores, min_val_loss = run_training(args, logger, return_val=True)
    print(f"*************** min_val_loss for trial {trial_number}: {min_val_loss} ***************")
    trial_dict = dict(vars(args))
    trial_dict["min_val_loss"] = min_val_loss
    trial_dict["test_metric"] = np.nanmean(ensemble_scores)
    trial_dict["hpo_space_source"] = hpo_space_source
    trial_dict["hpo_sampled_params"] = sampled_params
    with open(f"{args.save_dir}/params.json", "w") as outfile: 
        json.dump(trial_dict, outfile)

    # Move ckpt to actual path
    final_save_dir = os.path.join(parent_save_dir, f"trial_{trial_number}")
    print(f"Moving {args.save_dir} to {final_save_dir}")
    shutil.move(args.save_dir, final_save_dir)

    args.save_dir = parent_save_dir

    return min_val_loss


if __name__ == "__main__":

    args = parse_args()
    print(f"args: {args}")
    # setup random seed
    setup(seed=args.seed)

    # Set up Optuna storage
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{args.save_dir}/optuna.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="minimize", load_if_exists=True,
    )

    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    if args.parser_name != 'finetune':
        raise ValueError(f"Not HPO NYI for {args.parser_name} mode")
    
    if args.n_trials is None:
        raise ValueError(f"--n_trials cannot be None during HPO")

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)

    hpo_space, hpo_space_source = resolve_hpo_space(args)
    print(f"Resolved HPO search space from: {hpo_space_source}")
    
    print(f"Number of trials for HPO: {args.n_trials}")
    objective = partial(objective_all, args=args, logger=logger, hpo_space=hpo_space, hpo_space_source=hpo_space_source)
    study.optimize(objective, n_trials=args.n_trials, timeout=None,
                    callbacks=[MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE,))])

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
