"""Implement the coordinator node for a rooster simulation or experiment."""

import rclpy
import numpy as np
from auto_msgs2.msg import FromAutobox, ToAutobox
from casadi_tools.dynamics import integrators
from casadi_tools.simulation import simulator as sim
from models import single_track as st
from models import world as wd
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy import node, parameter, qos
from casadi_tools.nlp_utils import casadi_builder as cb

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
Note: In this simulation, the vehicle's motion is propagated in the global frame (East/North/Psi)
"""

_STATE_PARAM_NAMES = [
    "ux_mps",
    "uy_mps",
    "r_radps",
    "east_m",
    "north_m",
    "psi_rad",
    "dfz_long_kn",
    "dfz_lat_kn",
]
_INPUT_PARAM_NAMES = [
    "delta_rad",
    "fx_kn",
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
   
        self.fromauto_pub = self.create_publisher(
            msg_type=FromAutobox,
            topic="/simulator/from_autobox",
            qos_profile=_RELIABLE_PUBSUB_QOS,
        )

    def init_simulation(self, simulator: sim.SimRunner, init_inputs: st._Inputs) -> None:
        self.sim = simulator
        self.curr_states = self.sim.begin()
        self.curr_inputs = init_inputs
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
    ) -> tuple[st._StatesGlobal, st._Inputs, float]:
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

        sim_time_s = self.get_parameter("sim_time_s").get_parameter_value().double_value
        self.enable_mpc_bool = (
            self.get_parameter("enable_mpc_bool").get_parameter_value().double_value
        )

        init_states = st._StatesGlobal(
            ux_mps=state_params["ux_mps"],
            uy_mps=state_params["uy_mps"],
            r_radps=state_params["r_radps"],
            dfz_long_kn=0.1,
            dfz_lat_kn=0.1,
            east_m=state_params["east_m"],
            north_m=state_params["north_m"],
            psi_rad=state_params["psi_rad"],
        )

        return init_states, init_inputs, sim_time_s


    def pack_fromautobox(self) -> FromAutobox:
        
        return FromAutobox(
            heartbeat=self.sim.current_event,
            t_s=self.sim.current_time,
            pre_flag=1,
            ux_mps=self.curr_states.ux_mps,
            uy_mps=self.curr_states.uy_mps,
            r_radps=self.curr_states.r_radps,
            dfz_long_est_kn=self.curr_states.dfz_long_kn,
            dfz_lat_est_kn=self.curr_states.dfz_lat_kn,
            east_m=self.curr_states.east_m,
            north_m=self.curr_states.north_m,
            psi_rad=self.curr_states.psi_rad,
            s_m=-1717.0,
            e_m=-1717.0,
            dpsi_rad=-1717.0,
            delta_cmd_rad=self.curr_inputs.delta_rad,
            delta_meas_rad=0.0,
            fx_cmd_kn=self.curr_inputs.fx_kn,
            fx_meas_kn=0.0,
            user_def0=self.enable_mpc_bool,
            user_def1=0.0,
            user_def2=0.0,
            user_def3=0.0,
        )

    def timer_callback(self) -> None:
        try:
            self.curr_states = self.sim.take_step(inputs=self.curr_inputs, params=0.0)
        except StopIteration as err:
            raise EndOfSim from err
        else:
            self.fromauto_pub.publish(self.pack_fromautobox())
            

    def to_autobox_callback(self, msg: ToAutobox) -> None:
        self.curr_inputs.delta_rad = msg.delta_cmd_rad
        self.curr_inputs.fx_kn = msg.fx_cmd_kn

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
    init_states, init_inputs, sim_time_s = sim_node.get_initial_conditions(
        world=config.WORLD
    )

    integrator = integrators.create_integrator(
        integrator=integrators.euler,
        oracle=config.SIM_VEHICLE_MODEL.temporal_global_dynamics,
        num_states=st._StatesGlobal.num_fields,
        num_inputs=st._Inputs.num_fields,
    )
    simulator = sim.SimRunner.create_sim(
        integrator=integrator,
        end_time=sim_time_s,
        step=SIM_STEP_S,
        init_states=init_states,
    )

    sim_node.init_simulation(simulator=simulator, init_inputs=init_inputs)

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
