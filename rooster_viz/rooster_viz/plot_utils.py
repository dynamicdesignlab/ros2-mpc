""" Rosbag to rooster parsing module

This module contains functions and classes to convert rosbags to normal
python dictionaries compatible with rooster plotting functions

"""

import re
from typing import Dict, Tuple

import casadi as ca
import numpy as np
from models import world as wd
from rooster import mpc_formulation


def form_array_from_keys(
    in_data: Dict[str, np.ndarray], ordered_keys: Tuple[str], epoch_idx: int
) -> np.ndarray:
    out_dict = {key: in_data[key][epoch_idx] for key in ordered_keys}
    return np.row_stack(tuple(out_dict.values()))


def rowstack_arr_in_dict(
    in_dict: Dict[str, np.ndarray], data: np.ndarray, key: str
) -> Dict[str, np.ndarray]:
    try:
        in_dict[key] = np.row_stack((in_dict[key], data))
    except KeyError:
        in_dict[key] = data

    return in_dict




def parse_mpc_data(
    mpc_data: Dict[str, np.ndarray],
    world: wd.SimpleWorld,
    mpc_problem: mpc_formulation.MPCFormulation,
    stage_cost_params: mpc_formulation.StageObjParams,
    term_cost_params: mpc_formulation.TerminalObjParams,
) -> Dict[str, np.ndarray]:
    
    num_epoch, num_stage = mpc_data["s_m"].shape

    s_vec = mpc_data["s_m"].reshape(-1, 1)
    e_vec = mpc_data["e_m"].reshape(-1, 1)
    dpsi_vec = mpc_data["dpsi_rad"].reshape(-1, 1)

    east_m, north_m, psi_rad = world.seu_to_enu(s_vec, e_vec, dpsi_vec)

    mpc_data["east_m"] = east_m.reshape((num_epoch, num_stage))
    mpc_data["north_m"] = north_m.reshape((num_epoch, num_stage))
    mpc_data["psi_rad"] = psi_rad.reshape((num_epoch, num_stage))

    mpc_data["t0_s"] = mpc_data["t_s"][:, 0]    

    return mpc_data


def parse_data_for_plotting(
    mpc_data: Dict[str, np.ndarray],
    mpc_problem: mpc_formulation.MPCFormulation,
    stage_cost_params: mpc_formulation.StageObjParams,
    term_cost_params: mpc_formulation.TerminalObjParams,
    world: wd.SimpleWorld,
) -> Dict[str, np.ndarray]:
    
    mpc_data = parse_mpc_data(
        mpc_data=mpc_data,
        world=world,
        mpc_problem=mpc_problem,
        stage_cost_params=stage_cost_params,
        term_cost_params=term_cost_params,
    )

    return mpc_data
