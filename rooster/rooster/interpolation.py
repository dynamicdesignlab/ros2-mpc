import casadi as ca
import numpy as np
from models import world as wd


def get_interp_k_psi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    k_1pm = np.array(world.k_1pm)

    if double:
        double_s, double_k = world.double_field("k_1pm")
        return ca.interpolant("k_1pm", "linear", [double_s], double_k)

    return ca.interpolant("k_psi_1pm", "linear", [s_m], k_1pm)


def get_interp_ux(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    ux_des_mps = np.array(world.ux_des_mps)

    if double:
        double_s, double_ux = world.double_field("ux_des_mps")
        return ca.interpolant("ux_des_mps", "linear", [double_s], double_ux)

    return ca.interpolant("ux_des_mps", "linear", [s_m], ux_des_mps)


def get_interp_e_max(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    e_max_m = np.array(world.e_max_m)

    if double:
        double_s, double_e_max_m = world.double_field("e_max_m")
        return ca.interpolant("e_max_m", "linear", [double_s], double_e_max_m)

    return ca.interpolant("e_max_m", "linear", [s_m], e_max_m)
