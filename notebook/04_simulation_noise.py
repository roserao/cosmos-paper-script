"""
Parallel execution of simulations (amplification of noise parameters)
"""

from concurrent.futures import ProcessPoolExecutor
import os
import json
import pickle

from dataclasses import asdict
from itertools import product
from typing import Literal

import pandas as pd

from cosmos import Simulator, ModelAnalyzer
from cosmos.simulator import Config

SAVE_ROOT = os.path.expanduser("~/Desktop/cosmos-paper-script/test/sim_1000")

N_POS = 1000


def test_config(config: Config, name: str) -> None:

    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)

    res_path = os.path.join(save_dir, "compare.pkl")

    if os.path.exists(res_path):
        print(f"Results for {name} already exist. Skipping simulation.")
        return

    with open(f"{save_dir}/config.pkl", "wb") as f:
        pickle.dump(config, f)
    with open(f"{save_dir}/readable_config.json", "w") as f:
        json.dump(asdict(config), f)  # For human readability

    simulator = Simulator(config=config)
    simulator.simulate()
    simulator.build_cosmos(os.path.join(save_dir, "model"))

    for i in simulator.model.all_group_new_index:
        print(f"Running Cosmos on combined group {i}...", end="\r")
        simulator.model.run_cosmos(group_new_idx=i, no_s_hat=False)

    analyzer = ModelAnalyzer(
        model=simulator.model,
        data_path=os.path.join(save_dir, "data"),
    )

    with open(os.path.join(save_dir, "simulator.pkl"), "wb") as f:
        pickle.dump(simulator, f)

    with open(os.path.join(save_dir, "analyzer.pkl"), "wb") as f:
        pickle.dump(analyzer, f)

    df_best_models = analyzer.best_models
    df_position = simulator.df_position

    true_models = df_position.set_index("position")["model"]
    selected_models = df_best_models.set_index("position")["model_rank1"]

    compare_df = pd.DataFrame(
        {
            "true_model": true_models,
            "selected_model": selected_models,
        }
    )

    compare_df.to_pickle(res_path)


def get_noisy_config(
    n_pos: int,
    magnitude: float = 1.0,
    param: Literal["all", "x", "y", "theta"] = "all",
    seed: int = 1000,
) -> Config:

    config = Simulator.default_config()
    config.simulation.n_position = n_pos
    config.simulation.seed = seed % (2**32)

    match param:
        case "all":
            config.observation.sigma_x *= magnitude
            config.observation.sigma_y *= magnitude
            config.observation.sigma_theta *= magnitude
        case "x":
            config.observation.sigma_x *= magnitude
        case "y":
            config.observation.sigma_y *= magnitude
        case "theta":
            config.observation.sigma_theta *= magnitude

    return config


def task_noise(args):
    idx, (i, param) = args

    if i == 0 and param != "all":
        return  # i == 0 is equivalent to default noise, only need to run once

    magnitude = 2**i
    name = f"noise_{magnitude:d}_{param}"
    save_path = os.path.join(SAVE_ROOT, name, "compare.pkl")

    if os.path.exists(save_path):
        print(f"Skipping {name} as results already exist.")
        return

    seed = (idx + 5956) * 5243
    config = get_noisy_config(n_pos=N_POS, magnitude=magnitude, param=param, seed=seed)

    print(f"Testing config: {name} with magnitude {magnitude} for parameter {param}")
    test_config(config=config, name=name)


def task_var_per_pos(args):

    idx, (log_magnitude, n_var_per_pos) = args

    magnitude = 2**log_magnitude

    # n_var_per_pos == 20 is the default, so we only run for 10 and 15
    name = f"n_var_per_pos_{n_var_per_pos:d}_noise_{magnitude:d}"

    if os.path.exists(os.path.join(SAVE_ROOT, name, "compare.pkl")):
        print(f"Skipping {name} as results already exist.")
        return

    seed = (idx + 5109) * 5616

    config = get_noisy_config(n_pos=N_POS, magnitude=magnitude, param="all", seed=seed)

    config.simulation.n_variant_per_position = n_var_per_pos

    print(f"Testing config: {name} with n_var_per_pos {n_var_per_pos} and magnitude {magnitude:d}")
    test_config(config=config, name=name)


def main():
    args_list = list(enumerate(product(range(5), ["all", "x", "y", "theta"])))


    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(task_noise, args_list)

    executor.shutdown(wait=True)

    args_list = list(enumerate(product(range(5), [10, 15])))

    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(task_var_per_pos, args_list)

    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
