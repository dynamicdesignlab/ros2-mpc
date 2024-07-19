"""Implement MPC formulation for rooster path tracking problem."""

from dataclasses import dataclass
from typing import Callable, ClassVar

import casadi as ca
import numpy as np
from casadi_tools import types
from casadi_tools.dynamics import named_arrays as na
from casadi_tools.nlp_utils import casadi_builder as cb
from models import single_track as st
from models import world as wd
from python_data_parsers import units
from python_data_parsers.units import SI_PREFIX

NUM_STAGES = 60
STEP_SIZE  = 0.05 # seconds

@dataclass
class StageObjParams(na.NamedVector):
    """Stagewise objective parameters."""

    w_ux: na.NV_FIELD_TYPE
    w_e: na.NV_FIELD_TYPE
    w_dpsi: na.NV_FIELD_TYPE
    w_delta_dot: na.NV_FIELD_TYPE
    w_fx_dot: na.NV_FIELD_TYPE


@dataclass
class StageObjResult(na.NamedVector):
    """Stagewise objective terms."""

    J_ux: na.NV_FIELD_TYPE
    J_e: na.NV_FIELD_TYPE
    J_dpsi: na.NV_FIELD_TYPE
    J_delta_dot: na.NV_FIELD_TYPE
    J_fx_dot: na.NV_FIELD_TYPE


StageObjResultArray = na.NamedArray.create_from_namedvector(
    "StageObjResultArray", StageObjResult
)


@dataclass
class TerminalObjParams(na.NamedVector):
    """Teminal objective parameters."""

    w_terminal_ux: na.NV_FIELD_TYPE
    w_terminal_e: na.NV_FIELD_TYPE
    w_terminal_dpsi: na.NV_FIELD_TYPE


@dataclass(eq=False)
class TerminalObjResult(na.NamedVector):
    """Teminal objective terms."""

    J_terminal_ux: na.NV_FIELD_TYPE
    J_terminal_e: na.NV_FIELD_TYPE
    J_terminal_dpsi: na.NV_FIELD_TYPE


TerminalObjResultArray = na.NamedArray.create_from_namedvector(
    "TerminalObjResultArray", TerminalObjResult
)


@dataclass
class MPCStates(st._StatesPath):
    """States for vehicle in path reference frame, including actuators."""

    # Inherits all of the states from st.Model.StatesPath and adds the ones below
    delta_rad: na.NV_FIELD_TYPE
    fx_kn: na.NV_FIELD_TYPE


@dataclass
class MPCInputs(na.NamedVector):
    """Inputs for vehicle in MPC formulation."""

    delta_dot_radps: na.NV_FIELD_TYPE
    fx_dot_knps: na.NV_FIELD_TYPE


@cb.casadi_dataclass
class MPCFormulation:
    """
    MPC formulation class.

    Contains all the parameters and methods necessary to fully
    describe the MPC problem to be solved.

    """

    model: st.Model
    """Dynamic model."""
    interp_ux: Callable
    """Interpolant for desired longitudinal speed in terms of path progress [m/s]."""
    interp_k_psi: Callable
    """Interpolant for the centerline curvature in terms of path progress [1/m]."""
    
    num_stages: ClassVar[int] = NUM_STAGES
    step_size: ClassVar[float] = STEP_SIZE


    @cb.casadi_method((MPCStates.num_fields, MPCInputs.num_fields))
    def dynamics_with_kappa(self, states_vec, inputs_vec) -> types.CASADI_SYMBOLIC:
        """
        Vehicle dynamics function wrapper.
        """

        states = MPCStates.from_array(states_vec)
        inputs = MPCInputs.from_array(inputs_vec)

        k_psi_1pm = self.interp_k_psi(states.s_m)

        states_st = st._StatesPath(
            ux_mps=states.ux_mps,
            uy_mps=states.uy_mps,
            r_radps=states.r_radps,
            dfz_long_kn=states.dfz_long_kn,
            dfz_lat_kn=states.dfz_lat_kn,
            s_m=states.s_m,
            e_m=states.e_m,
            dpsi_rad=states.dpsi_rad,
        )

        inputs_st = st._Inputs(
            delta_rad=states.delta_rad,
            fx_kn=states.fx_kn,
        )

        states_st_vec = states_st.to_array()
        inputs_st_vec = inputs_st.to_array()

        dstates_st_vec = self.model.temporal_path_dynamics(
            states_st_vec, inputs_st_vec, k_psi_1pm,
        )

        dstates_st = st._StatesPath.from_array(dstates_st_vec)

        dstates = MPCStates(
            ux_mps=dstates_st.ux_mps,
            uy_mps=dstates_st.uy_mps,
            r_radps=dstates_st.r_radps,
            dfz_long_kn=dstates_st.dfz_long_kn,
            dfz_lat_kn=dstates_st.dfz_lat_kn,
            s_m=dstates_st.s_m,
            e_m=dstates_st.e_m,
            dpsi_rad=dstates_st.dpsi_rad,
            delta_rad=inputs.delta_dot_radps,
            fx_kn=inputs.fx_dot_knps,
        )

        return dstates.to_array()

    @cb.casadi_method((MPCStates.num_fields, MPCInputs.num_fields, StageObjParams.num_fields,), num_outputs=2,)
    def stage_objective(
        self,
        states_vec,
        inputs_vec,
        stage_params_vec,
    ):
        """
        Calculate the stagewise objective value.
        """

        states = MPCStates.from_array(states_vec)
        inputs = MPCInputs.from_array(inputs_vec)
        stage_params = StageObjParams.from_array(stage_params_vec)


        ux_des_mps = self.interp_ux(states.s_m)

        delta_ux = states.ux_mps - ux_des_mps
        J_ux = stage_params.w_ux * (delta_ux**2)

        J_e  = stage_params.w_e * (states.e_m**2)
        J_dpsi = stage_params.w_dpsi * (states.dpsi_rad**2)

        J_delta_dot = stage_params.w_delta_dot * (inputs.delta_dot_radps**2)
        J_fx_dot = stage_params.w_fx_dot * (inputs.fx_dot_knps**2)

        J_items = StageObjResult(
            J_ux=J_ux,
            J_e=J_e,
            J_dpsi=J_dpsi,
            J_delta_dot=J_delta_dot,
            J_fx_dot=J_fx_dot,
        )

        # Total cost
        J_tot = (
            J_ux
            + J_e
            + J_dpsi
            + J_ux
            + J_delta_dot
            + J_fx_dot
        )

        return J_tot, J_items.to_array()

    @cb.casadi_method((MPCStates.num_fields, TerminalObjParams.num_fields), num_outputs=2)
    def terminal_objective(
        self,
        states_vec: types.CASADI_SYMBOLIC,
        terminal_params_vec: types.CASADI_SYMBOLIC,
    ):
        """
        Calculate the terminal objective value.
        """

        states = MPCStates.from_array(states_vec)
        terminal_params = TerminalObjParams.from_array(terminal_params_vec)

        ux_des_mps = self.interp_ux(states.s_m)

        J_ux_term = terminal_params.w_terminal_ux * (states.ux_mps - ux_des_mps) ** 2
        J_e_term = terminal_params.w_terminal_e * states.e_m**2
        J_dpsi_term = terminal_params.w_terminal_dpsi * states.dpsi_rad**2

        J_item = TerminalObjResult(
            J_terminal_ux=J_ux_term,
            J_terminal_e=J_e_term,
            J_terminal_dpsi=J_dpsi_term,
        )

        J_tot = J_ux_term + J_e_term + J_dpsi_term

        return J_tot, J_item.to_array()
    
    @cb.casadi_method((MPCStates.num_fields,), num_outputs=1)
    def power_inequality(self, states_vec: types.CASADI_SYMBOLIC) -> types.CASADI_SYMBOLIC:
        """
        Calculate the inequality expressions for front axle power.
        """
        
        states = MPCStates.from_array(states_vec)
        delta_rad = states.delta_rad
        fx_kn = states.fx_kn

        st_states = st._StatesVelocity(
            ux_mps=states.ux_mps,
            uy_mps=states.uy_mps,
            r_radps=states.r_radps,
        )
        inputs = st._Inputs(delta_rad=delta_rad, fx_kn=fx_kn)

        power_ub_kW = self.model.power_limit(st_states.to_array(), inputs.to_array())

        return power_ub_kW