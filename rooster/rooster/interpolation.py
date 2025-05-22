import casadi as ca
import numpy as np
from models import world as wd


def get_interp_psi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    psi_cl_rad = np.array(world.psi_cl_rad)

    if double:
        double_s, double_psi = world.double_field("psi_cl_rad")
        return ca.interpolant("psi_cl_rad", "linear", [double_s], double_psi)

    return ca.interpolant("psi_cl_rad", "linear", [s_m], psi_cl_rad)


def get_interp_phi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    phi_cl_rad = np.array(world.phi_cl_rad)

    if double:
        double_s, double_phi = world.double_field("phi_cl_rad")
        return ca.interpolant("phi_cl_rad", "linear", [double_s], double_phi)

    return ca.interpolant("phi_cl_rad", "linear", [s_m], phi_cl_rad)


def get_interp_theta(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    theta_cl_rad = np.array(world.theta_cl_rad)

    if double:
        double_s, double_theta = world.double_field("theta_cl_rad")
        return ca.interpolant("theta_cl_rad", "linear", [double_s], double_theta)

    return ca.interpolant("theta_cl_rad", "linear", [s_m], theta_cl_rad)


def get_interp_k_psi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    k_psi = np.array(world.k_psi_cl_radpm)

    if double:
        double_s, double_k = world.double_field("k_psi_cl_radpm")
        return ca.interpolant("k_psi_cl_radpm", "linear", [double_s], double_k)

    return ca.interpolant("k_psi_cl_radpm", "linear", [s_m], k_psi)


def get_interp_k_phi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    k_phi = np.array(world.k_phi_cl_radpm)

    if double:
        double_s, double_k = world.double_field("k_phi_cl_radpm")
        return ca.interpolant("k_phi_cl_radpm", "linear", [double_s], double_k)

    return ca.interpolant("k_phi_cl_radpm", "linear", [s_m], k_phi)


def get_interp_k_theta(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    k_theta = np.array(world.k_theta_cl_radpm)

    if double:
        double_s, double_k = world.double_field("k_theta_cl_radpm")
        return ca.interpolant("k_theta_cl_radpm", "linear", [double_s], double_k)

    return ca.interpolant("k_theta_cl_radpm", "linear", [s_m], k_theta)


def get_interp_ux(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    ux_ref_mps = np.array(world.ux_ref_mps)

    if double:
        double_s, double_ux = world.double_field("ux_ref_mps")
        return ca.interpolant("ux_ref_mps", "linear", [double_s], double_ux)

    return ca.interpolant("ux_ref_mps", "linear", [s_m], ux_ref_mps)

def get_interp_e(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    e_ref_m = np.array(world.e_ref_m)

    if double:
        double_s, double_e = world.double_field("e_ref_m")
        return ca.interpolant("e_ref_m", "linear", [double_s], double_e)

    return ca.interpolant("e_ref_m", "linear", [s_m], e_ref_m)

def get_interp_dpsi(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    dpsi_ref_rad = np.array(world.dpsi_ref_rad)

    if double:
        double_s, dpsi_ref_rad = world.double_field("dpsi_ref_rad")
        return ca.interpolant("dpsi_ref_rad", "linear", [double_s], dpsi_ref_rad)

    return ca.interpolant("dpsi_ref_rad", "linear", [s_m], dpsi_ref_rad)


def get_interp_e_max(world: wd.SimpleWorld, double: bool = False):
    s_m = np.array(world.s_m)
    e_max_m = np.array(world.e_max_m)

    if double:
        double_s, double_e_max_m = world.double_field("e_max_m")
        return ca.interpolant("e_max_m", "linear", [double_s], double_e_max_m)

    return ca.interpolant("e_max_m", "linear", [s_m], e_max_m)
