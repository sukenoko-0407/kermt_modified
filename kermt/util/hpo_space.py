# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for loading and validating HPO search space definitions.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple, Any

REQUIRED_PARAM_NAMES = {
    "max_lr",
    "final_lr_factor",
    "dropout",
    "attn_out",
    "dist_coff",
    "fine_tune_coff",
    "bond_drop_rate",
    "ffn_num_layers",
    "ffn_hidden_size",
}

DEFAULT_HPO_SPACE = {
    "init_lr_factor": 10,
    "params": {
        "max_lr": {"type": "float", "low": 1e-4, "high": 1e-3, "step": 2e-4},
        "final_lr_factor": {"type": "int", "low": 2, "high": 10, "step": 2},
        "dropout": {"type": "categorical", "choices": [0, 0.05, 0.1, 0.2]},
        "attn_out": {"type": "int", "low": 4, "high": 8, "step": 4},
        "dist_coff": {"type": "float", "low": 0.05, "high": 0.15, "step": 0.05},
        "fine_tune_coff": {"type": "categorical", "choices": [1.0]},
        "bond_drop_rate": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.2},
        "ffn_num_layers": {"type": "int", "low": 2, "high": 3, "step": 1},
        "ffn_hidden_size": {"type": "int", "low": 700, "high": 1300, "step": 600},
    },
}

PROFILE_TO_CONFIG_PATH = {
    "small": "configs/hpo/finetune_small.json",
    "medium": "configs/hpo/finetune_medium.json",
    "large": "configs/hpo/finetune_large.json",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_spec(name: str, spec: Dict[str, Any]) -> None:
    if "type" not in spec:
        raise ValueError(f'Parameter "{name}" must include "type".')

    spec_type = spec["type"]
    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f'Parameter "{name}" categorical choices must be a non-empty list.')
        return

    if spec_type not in {"float", "int"}:
        raise ValueError(f'Parameter "{name}" has unsupported type "{spec_type}".')

    low = spec.get("low")
    high = spec.get("high")
    if low is None or high is None:
        raise ValueError(f'Parameter "{name}" must include both "low" and "high".')
    if low > high:
        raise ValueError(f'Parameter "{name}" has low > high ({low} > {high}).')

    log = bool(spec.get("log", False))
    step = spec.get("step")
    if log and step is not None:
        raise ValueError(f'Parameter "{name}" cannot use both "log" and "step".')

    if spec_type == "int" and step is not None and step <= 0:
        raise ValueError(f'Parameter "{name}" int step must be positive.')

    if spec_type == "float" and step is not None and step <= 0:
        raise ValueError(f'Parameter "{name}" float step must be positive.')


def validate_hpo_space(space: Dict[str, Any]) -> None:
    if not isinstance(space, dict):
        raise ValueError("HPO space must be a JSON object.")

    init_lr_factor = space.get("init_lr_factor", None)
    if init_lr_factor is None or init_lr_factor <= 0:
        raise ValueError('"init_lr_factor" must be provided and greater than 0.')

    params = space.get("params")
    if not isinstance(params, dict):
        raise ValueError('"params" must be a JSON object.')

    missing = REQUIRED_PARAM_NAMES - set(params.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"HPO space is missing required params: {missing_str}")

    for name, spec in params.items():
        if not isinstance(spec, dict):
            raise ValueError(f'Parameter "{name}" spec must be an object.')
        _validate_spec(name=name, spec=spec)


def resolve_hpo_space(args: Namespace) -> Tuple[Dict[str, Any], str]:
    """
    Resolves and validates HPO space from CLI options.

    :return: Tuple of (search_space_dict, source_description)
    """
    if args.hpo_profile is not None and args.hpo_config_path is not None:
        raise ValueError('Only one of --hpo_profile and --hpo_config_path can be specified.')

    if args.hpo_profile is None and args.hpo_config_path is None:
        space = DEFAULT_HPO_SPACE
        source = "default (main_hpo.py legacy range)"
    elif args.hpo_config_path is not None:
        config_path = Path(args.hpo_config_path).expanduser().resolve()
        if not config_path.exists():
            raise ValueError(f'HPO config file does not exist: "{config_path}"')
        space = _load_json(config_path)
        source = str(config_path)
    else:
        rel_path = PROFILE_TO_CONFIG_PATH[args.hpo_profile]
        config_path = _repo_root() / rel_path
        if not config_path.exists():
            raise ValueError(f'Built-in HPO profile config not found: "{config_path}"')
        space = _load_json(config_path)
        source = f'{args.hpo_profile} ({config_path})'

    validate_hpo_space(space)
    return space, source


def suggest_from_space(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Samples parameters from a validated search space definition.
    """
    sampled = {}
    for name, spec in space["params"].items():
        spec_type = spec["type"]
        if spec_type == "categorical":
            sampled[name] = trial.suggest_categorical(name, choices=spec["choices"])
        elif spec_type == "int":
            kwargs = {}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            if "log" in spec:
                kwargs["log"] = bool(spec["log"])
            sampled[name] = trial.suggest_int(name, spec["low"], spec["high"], **kwargs)
        elif spec_type == "float":
            kwargs = {}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            if "log" in spec:
                kwargs["log"] = bool(spec["log"])
            sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
        else:
            raise ValueError(f'Unsupported spec type "{spec_type}" for "{name}".')
    return sampled
