"""Create casadi solver from MPC formulation."""

import casadi as ca
import numpy as np
from casadi_tools.dynamics import integrators as integ
from casadi_tools.dynamics import projector as proj
from casadi_tools.nlp_utils import nlp_problem as nlp
from models import single_track as st
from models import vehicle_params as vp
from casadi_tools.nlp_utils import casadi_builder as cb

from rooster import config
from rooster import mpc_formulation as mpc


def add_stagewise_objective(
    nlp_in: nlp.NLPProblem, mpc_model: mpc.MPCFormulation
) -> nlp.NLPProblem:

    nlp_in.params.add_field("stage_cost", mpc.StageObjParams.num_fields)
    rep_costs = ca.repmat(nlp_in.params.get_expr("stage_cost"), 1, nlp_in.num_stages(0))
    obj_map = mpc_model.stage_objective.map(nlp_in.num_stages(0))

    obj_val_vector, _ = obj_map(
        nlp_in.states.to_array(),
        nlp_in.inputs.to_array(),
        rep_costs,
    )

    nlp_in.add_objective(ca.sum2(obj_val_vector))

    return nlp_in


def add_terminal_objective(
    nlp_in: nlp.NLPProblem, mpc_model: mpc.MPCFormulation
) -> nlp.NLPProblem:
    nlp_in.params.add_field("term_cost", mpc.TerminalObjParams.num_fields)

    J_term, _ = mpc_model.terminal_objective(
        nlp_in.states.to_array()[:, -1], nlp_in.params.get_expr("term_cost")
    )

    nlp_in.add_objective(J_term)

    return nlp_in


def add_power_constraints(nlp_in: nlp.NLPProblem, mpc_model: mpc.MPCFormulation
) -> nlp.NLPProblem:
    power_lim = mpc_model.power_inequality.map(nlp_in.num_stages(0))
    power_ub = power_lim(nlp_in.states.to_array())
    nlp_in.cstrs.add_field("power_ub", power_ub, upper=0.0)
    return nlp_in


def set_box_constraints(
    nlp_in: nlp.NLPProblem, veh_params: vp.VehicleParams
) -> nlp.NLPProblem:
    
    max_delta_rad = np.radians(veh_params.max_delta_deg)
    max_delta_dot_radps = np.radians(veh_params.max_delta_dot_degps)
    nlp_in.states.set_bounds("delta_rad", lower=-max_delta_rad, upper=max_delta_rad)
    nlp_in.inputs.set_bounds("delta_dot_radps", lower=-max_delta_dot_radps, upper=max_delta_dot_radps)


    min_fx_kn = veh_params.min_fx_kn
    min_fx_dot_knps = veh_params.min_fx_dot_knps
    max_fx_kn = veh_params.max_fx_kn
    max_fx_dot_knps = veh_params.max_fx_dot_knps
    nlp_in.states.set_bounds("fx_kn", lower=min_fx_kn, upper=max_fx_kn)
    nlp_in.inputs.set_bounds("fx_dot_knps", lower=min_fx_dot_knps, upper=max_fx_dot_knps)

    return nlp_in


def print_info() -> None:
    print(f"\n\n{'#'*80}")
    print("ROOSTER NMPC GENERATION")
    print(f"{'#'*80}\n")

    header_str = "Parameters:"
    print(header_str)
    print(f"{'-'*len(header_str)}")
    print(f"\tN: {mpc.NUM_STAGES}")
    print(f"\tdt: {mpc.STEP_SIZE}")
    print(f"\tReplan Time: {config.REPLAN_TIME}")
    print(f"\tWorld: {config.WORLD_FILE.stem}")
    print(f"\tVehicle: {config.VEHICLE_PARAMS.name}")
    print(f"\tFront Tire: {config.FRONT_TIRE}")
    print(f"\tRear Tire: {config.REAR_TIRE}\n")



def main():
    print_info()
    veh_integ = integ.create_integrator(
        integrator=integ.trapz,
        oracle=config.MPC_PROBLEM.dynamics_with_kappa,
        num_states=mpc.MPCStates.num_fields,
        num_inputs=mpc.MPCInputs.num_fields,
    )

    horizon = nlp.constant_horizon(num_stages=config.MPC_PROBLEM.num_stages, step_size=config.MPC_PROBLEM.step_size)

    nlp_in = nlp.create_nlp_with_dynamics(
        state_names=mpc.MPCStates.field_names,
        input_names=mpc.MPCInputs.field_names,
        dynamics=veh_integ,
        horizon=horizon,
    )

    nlp_in = set_box_constraints(nlp_in=nlp_in, veh_params=config.VEHICLE_PARAMS)
    nlp_in = add_stagewise_objective(nlp_in=nlp_in, mpc_model=config.MPC_PROBLEM)
    nlp_in = add_terminal_objective(nlp_in=nlp_in, mpc_model=config.MPC_PROBLEM)
    nlp_in = add_power_constraints(nlp_in=nlp_in, mpc_model=config.MPC_PROBLEM)


    # Build solver shared object and pickle nlp description

    print("")
    solver = nlp.generate_solver(
        name=config.MPC_SOLVER_NAME,
        nlp=nlp_in,
        opt_type=nlp.OptType.NLP,
        solver="ipopt",
        opts={"ipopt": {"linear_solver": "ma57"}},
    )
    nlp.generate_shared_object(
        solver=solver,
        nlp=nlp_in,
        output_dir=config.CODEGEN_DIR,
        output_name=config.MPC_SOLVER_NAME,
        opt_level=config.MPC_GEN_OPT_LEVEL,
    )


if __name__ == "__main__":
    main()
