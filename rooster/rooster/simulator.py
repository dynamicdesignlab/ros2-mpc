"""Implement the coordinator node for a rooster simulation or experiment."""

import rclpy
import numpy as np
from auto_msgs2.msg import FromAutobox, ToAutobox
from casadi_tools.dynamics import integrators
from casadi_tools.simulation import simulator as sim
from models import nn_dynamics as st
#from models import single_track as st
from models import world as wd
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy import node, parameter, qos
from casadi_tools.nlp_utils import casadi_builder as cb
from casadi_tools import types
from typing import Callable, ClassVar

from rooster import config, interpolation

SIM_STEP_S = 0.01

_RELIABLE_PUBSUB_QOS = qos.QoSProfile(
    reliability=qos.QoSReliabilityPolicy.RELIABLE,
    durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=qos.QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
_BESTEFFORT_PUBSUB_QOS = qos.QoSProfile(
    reliability=qos.QoSReliabilityPolicy.BEST_EFFORT,
    durability=qos.QoSDurabilityPolicy.VOLATILE,
    history=qos.QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

"""
Note: In this simulation, the vehicle's motion is propagated in the path-relative frame (s/e/b/dpsi)
"""

_STATE_PARAM_NAMES = [
    "ux_mps",
    "uy_mps",
    "r_radps",
    "dfz_long_kn",
    "dfz_lat_kn",
    "s_m",
    "e_m",
    "dpsi_rad",
]
_INPUT_PARAM_NAMES = [
    "delta_rad",
    "fx_kn",
]

_PAST_PARAM_NAMES = [
    "r_2",
    "r_1",
    "uy_2",
    "uy_1",
    "ux_2",
    "ux_1",
    "delta_2",
    "delta_1",
    "fx_2",
    "fx_1",
]

class EndOfSim(Exception):
    pass


class SimulatorNode(node.Node):
    def __init__(self, model: st.Model) -> None:
        super().__init__("simulator")
        self.model = model

        self.sim = None
        self.curr_states = None
        self.curr_inputs = None
        self.curr_past = None
   
        self.fromauto_pub = self.create_publisher(
            msg_type=FromAutobox,
            topic="/simulator/from_autobox",
            qos_profile=_RELIABLE_PUBSUB_QOS,
        )   

    def init_simulation(self, simulator: sim.SimRunner, init_inputs: st._Inputs, init_past: st._StatesPast) -> None:
        self.sim = simulator
        self.curr_states = self.sim.begin()
        self.curr_inputs = init_inputs
        self.curr_past = init_past
        self.fromauto_pub.publish(self.pack_fromautobox())

        self.toauto_sub = self.create_subscription(
            msg_type=ToAutobox,
            topic="/simulator/to_autobox",
            callback=self.to_autobox_callback,
            qos_profile=_RELIABLE_PUBSUB_QOS,
        )
        self.timer = self.create_timer(
            timer_period_sec=SIM_STEP_S, callback=self.timer_callback
        )

    def get_initial_conditions(
        self, world: wd.SimpleWorld,
    ) -> tuple[st._StatesPath, st._Inputs, float]:
        
        self.declare_parameters(
            namespace="init_inputs",
            parameters=[
                (name, parameter.Parameter.Type.DOUBLE) for name in _INPUT_PARAM_NAMES
            ],
        )
        self.declare_parameters(
            namespace="init_states",
            parameters=[
                (name, parameter.Parameter.Type.DOUBLE) for name in _STATE_PARAM_NAMES
            ],
        )
        self.declare_parameters(
            namespace="init_past",
            parameters=[
                (name, parameter.Parameter.Type.DOUBLE) for name in _PAST_PARAM_NAMES
            ],
        )
        self.declare_parameter(
            name="sim_time_s",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            name="enable_mpc_bool",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )

        input_params = {
            key: self.extract_float_from_param(value)
            for key, value in self.get_parameters_by_prefix("init_inputs").items()
        }
        init_inputs = st._Inputs(**input_params)

        state_params = {
            key: self.extract_float_from_param(value)
            for key, value in self.get_parameters_by_prefix("init_states").items()
        }

        past_params = {
            key: self.extract_float_from_param(value)
            for key, value in self.get_parameters_by_prefix("init_past").items()
        }

        sim_time_s = self.get_parameter("sim_time_s").get_parameter_value().double_value
        self.enable_mpc_bool = (
            self.get_parameter("enable_mpc_bool").get_parameter_value().double_value
        )

        init_states = st._StatesPath(
            ux_mps=state_params["ux_mps"],
            uy_mps=state_params["uy_mps"],
            r_radps=state_params["r_radps"],
            dfz_long_kn=0.1,
            dfz_lat_kn=0.1,
            s_m=state_params["s_m"],
            e_m=state_params["e_m"],
            dpsi_rad=state_params["dpsi_rad"],
        )

        init_past = st._StatesPast(
            r_2 = past_params["r_2"],
            r_1 = past_params["r_1"],
            uy_2 = past_params["uy_2"],
            uy_1 = past_params["uy_1"],
            ux_2 = past_params["ux_2"],
            ux_1 = past_params["ux_1"],
            delta_2 = past_params["delta_2"],
            delta_1 = past_params["delta_1"],
            fx_2 = past_params["fx_2"],
            fx_1 = past_params["fx_1"],
        )

        return init_states, init_inputs, init_past, sim_time_s


    def pack_fromautobox(self) -> FromAutobox:

        east_m, north_m, _, psi_rad = config.WORLD.sebdpsi_to_enupsi(s_m=self.curr_states.s_m, e_m=self.curr_states.e_m, b_m=0.0, dpsi_rad=self.curr_states.dpsi_rad,)

        return FromAutobox(
            heartbeat=self.sim.current_event,
            t_s=self.sim.current_time,
            pre_flag=1,
            ux_mps=self.curr_states.ux_mps,
            uy_mps=self.curr_states.uy_mps,
            r_radps=self.curr_states.r_radps,
            dfz_long_est_kn=self.curr_states.dfz_long_kn,
            dfz_lat_est_kn=self.curr_states.dfz_lat_kn,
            east_m=east_m,
            north_m=north_m,
            psi_rad=psi_rad,
            s_m=self.curr_states.s_m,
            e_m=self.curr_states.e_m,
            dpsi_rad=self.curr_states.dpsi_rad,
            delta_cmd_rad=self.curr_inputs.delta_rad,
            fx_cmd_kn=self.curr_inputs.fx_kn,
            r_2=self.curr_past.r_2,
            r_1=self.curr_past.r_1,
            uy_2=self.curr_past.uy_2,
            uy_1=self.curr_past.uy_1,
            ux_2=self.curr_past.ux_2,
            ux_1=self.curr_past.ux_1,
            delta_2=self.curr_past.delta_2,
            delta_1=self.curr_past.delta_1,
            fx_2=self.curr_past.fx_2,
            fx_1=self.curr_past.fx_1,
        )

    def timer_callback(self) -> None:
        try:
            r_2_new = self.curr_past.r_1
            r_1_new = self.curr_states.r_radps
            uy_2_new = self.curr_past.uy_1
            uy_1_new = self.curr_states.uy_mps
            ux_2_new = self.curr_past.ux_1
            ux_1_new = self.curr_states.ux_mps
            self.curr_states = self.sim.take_step(inputs=self.curr_inputs, past=self.curr_past, params=0.0)
            self.curr_past.r_2 = r_2_new
            self.curr_past.r_1 = r_1_new
            self.curr_past.uy_2 = uy_2_new
            self.curr_past.uy_1 = uy_1_new
            self.curr_past.ux_2 = ux_2_new
            self.curr_past.ux_1 = ux_1_new
        except StopIteration as err:
            raise EndOfSim from err
        else:
            self.fromauto_pub.publish(self.pack_fromautobox())
            

    def to_autobox_callback(self, msg: ToAutobox) -> None:
        delta_2_new = self.curr_past.delta_1
        delta_1_new = self.curr_inputs.delta_rad
        self.curr_inputs.delta_rad = msg.delta_cmd_rad
        self.curr_past.delta_2 = delta_2_new
        self.curr_past.delta_1 = delta_1_new

        fx_2_new = self.curr_past.fx_1
        fx_1_new = self.curr_inputs.fx_kn
        self.curr_inputs.fx_kn = msg.fx_cmd_kn
        self.curr_past.fx_2 = fx_2_new
        self.curr_past.fx_1 = fx_1_new

    @staticmethod
    def extract_float_from_param(param: parameter.Parameter) -> float:
        return param.get_parameter_value().double_value
    



def main(args=None):
    """
    Entrypoint for ros2 executable.

    Instantiates a CoordinatorNode object and spins until shutdown.
    """

    rclpy.init(args=args)
    sim_node = SimulatorNode(model=config.SIM_VEHICLE_MODEL)
    init_states, init_inputs, init_past, sim_time_s = sim_node.get_initial_conditions(
        world=config.WORLD
    )

    @cb.casadi_function((st._StatesPath.num_fields, st._Inputs.num_fields, st._StatesPast.num_fields))
    def dynamics_with_track_curvature(states_vec, inputs_vec, past_vec):

        states = st._StatesPath.from_array(states_vec)

        psi_cl_rad       = config.MPC_PROBLEM.interp_psi(states.s_m)
        theta_cl_rad     = config.MPC_PROBLEM.interp_theta(states.s_m)
        phi_cl_rad       = config.MPC_PROBLEM.interp_phi(states.s_m)
        k_psi_cl_radpm   = config.MPC_PROBLEM.interp_k_psi(states.s_m)
        k_theta_cl_radpm = config.MPC_PROBLEM.interp_k_theta(states.s_m)
        k_phi_cl_radpm   = config.MPC_PROBLEM.interp_k_phi(states.s_m)

        track_curvature = st._TrackCurvature(
            psi_cl_rad=psi_cl_rad,
            theta_cl_rad=theta_cl_rad,
            phi_cl_rad=phi_cl_rad,
            k_psi_cl_radpm=k_psi_cl_radpm,
            k_theta_cl_radpm=k_theta_cl_radpm,
            k_phi_cl_radpm=k_phi_cl_radpm,
        )

        track_curvature_vec = track_curvature.to_array()

        return config.SIM_VEHICLE_MODEL.temporal_path_dynamics(states_vec, inputs_vec, track_curvature_vec, past_vec)


    integrator = integrators.create_integrator(
        integrator=integrators.euler,
        oracle=dynamics_with_track_curvature,
        num_states=st._StatesPath.num_fields,
        num_inputs=st._Inputs.num_fields,
        num_past = st._StatesPast.num_fields,
    )
    simulator = sim.SimRunner.create_sim(
        integrator=integrator,
        end_time=sim_time_s,
        step=SIM_STEP_S,
        init_states=init_states,
    )

    sim_node.init_simulation(simulator=simulator, init_inputs=init_inputs, init_past=init_past)

    try:
        rclpy.spin(node=sim_node)
    except KeyboardInterrupt:
        pass
    except EndOfSim:
        pass
    finally:
        sim_node.destroy_node()
        exit()


if __name__ == "__main__":
    main()
