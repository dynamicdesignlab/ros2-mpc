""" Rosbag to rooster parsing module

This module contains functions and classes to convert rosbags to normal
python dictionaries compatible with rooster plotting functions

"""

import pickle
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np
from casadi_tools.math import wrap_to_pi_float

from rooster import config
from rooster_viz import plot_utils, rosbag_parser


class ROSPlotData(NamedTuple):
    to_data: dict[str, np.ndarray]
    from_data: dict[str, np.ndarray]
    mpc_data: dict[str, np.ndarray]

    def asdict(self) -> dict[str, Any]:
        return {
            "to_data": self.to_data,
            "from_data": self.from_data,
            "mpc_data": self.mpc_data,
        }


def extract_from_data(
    ros_data: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    from_data = ros_data["/auto_bridge2/from_autobox"]
    from_data["psi_rad"] = wrap_to_pi_float(from_data["psi_rad"])

    return from_data


def extract_to_data(
    ros_data: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    to_data = ros_data["/auto_bridge2/to_autobox"]

    return to_data


def extract_mpc_data(ros_data: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    mpc_dict = ros_data["/rooster/nlp_out"]
    out_dict = {}

    for key, val in mpc_dict.items():
        if isinstance(val, dict):
            out_dict.update(val)
        else:
            out_dict[key] = val

    return out_dict


def parse_rosbag_for_plotting(rosbag_path: Path) -> ROSPlotData:
    pickle_path = rosbag_path.with_suffix(".pkl")
    if pickle_path.exists():
        with open(pickle_path, "rb") as pkl_file:
            plot_data = pickle.load(pkl_file)

        return ROSPlotData(**plot_data)

    bag_data = rosbag_parser.parse_rosbag(rosbag_path)

    from_data = extract_from_data(bag_data)
    to_data = extract_to_data(bag_data)
    mpc_data = extract_mpc_data(bag_data)

    mpc_plot_data = plot_utils.parse_data_for_plotting(
        mpc_data=mpc_data,
        stage_cost_params=config.STAGE_OBJ_PARAMS,
        term_cost_params=config.TERM_OBJ_PARAMS,
        world=config.WORLD,
        mpc_problem=config.MPC_PROBLEM,
    )

    plot_data = ROSPlotData(
        to_data=to_data,
        from_data=from_data,
        mpc_data=mpc_plot_data,
    )

    with open(pickle_path, "wb") as pkl_file:
        pickle.dump(plot_data.asdict(), pkl_file)

    return plot_data