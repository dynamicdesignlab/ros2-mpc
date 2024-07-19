import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from auto_msgs2.msg import FromAutobox
from casadi_tools.dynamics import named_arrays as na
from casadi_tools.nlp_utils import nlp_runner
from models import single_track as st
from rclpy import node, qos
from rooster_msgs.msg import NLPOutput, NLPResult, NLPSetup
from rooster_msgs.srv import NLPService

from rooster import config
from rooster import mpc_formulation as mpc
from rooster import utils

CODEGEN_DIR = (
    Path()
    .home()
    .joinpath("Workspace")
    .joinpath("src")
    .joinpath("rooster")
    .joinpath("rooster")
    .joinpath("codegen")
)
SOLVER_PATH = CODEGEN_DIR.joinpath("rooster_nlp.so")

StateArray = na.NamedArray.create_from_field_names(
    cls_name="StateArray", field_names=["ux_mps", "uy_mps", "r_radps", "dfz_long_kn", "dfz_lat_kn", "s_m", "e_m", "dpsi_rad"],
)

CmdArray = na.NamedArray.create_from_field_names(
    cls_name="CmdArray", field_names=["delta_rad", "fx_kn"],
)

CmdSlewArray = na.NamedArray.create_from_field_names(
    cls_name="CmdSlewArray", field_names=["delta_dot_radps", "fx_dot_knps"]
)

_SERV_QOS = qos.QoSProfile(
    durability=qos.QoSDurabilityPolicy.VOLATILE,
    reliability=qos.QoSReliabilityPolicy.RELIABLE,
    history=qos.QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
_PUBSUB_QOS = qos.QoSProfile(
    reliability=qos.QoSReliabilityPolicy.RELIABLE,
    durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=qos.QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass
class MPCRunner:
    """Implements MPCRunner class for calling nlpsol object with appropriate inputs."""

    nlp_run: nlp_runner.NLPRunner
    """NLPRunner object containing nlp solver."""

    # Arrays corresponding to the most recent horizon
    # These get updated when the NLP returns a usable solution
    _t_array: np.ndarray = field(init=False, repr=False)
    _s_array: np.ndarray = field(init=False, repr=False)
    _state_array: StateArray = field(init=False, repr=False)
    _input_array: CmdArray = field(init=False, repr=False)
    _input_slew_array: CmdSlewArray = field(init=False, repr=False)

    def __post_init__(self) -> None:

        self._first_solve = True
        ux_guess = 2.5 # m/s, initialization for first horizon

        self._s_array = np.linspace(0.0, 1e6, mpc.NUM_STAGES, endpoint=False)
        self._t_array = np.linspace(0.0, 1e6, mpc.NUM_STAGES, endpoint=False)

        self._state_array = StateArray.new_with_validation(
            ux_mps=ux_guess*np.ones((mpc.NUM_STAGES,)),
            uy_mps=np.zeros((mpc.NUM_STAGES,)),
            r_radps=np.zeros((mpc.NUM_STAGES,)),
            dfz_long_kn=0.1*np.ones((mpc.NUM_STAGES,)),
            dfz_lat_kn=0.1*np.ones((mpc.NUM_STAGES,)),
            s_m=np.linspace(0.0, 1e6, mpc.NUM_STAGES, endpoint=False),
            e_m=np.zeros((mpc.NUM_STAGES,)),
            dpsi_rad=np.zeros((mpc.NUM_STAGES,)),
        )

        self._input_array = CmdArray.new_with_validation(
            delta_rad=np.zeros((mpc.NUM_STAGES,)),
            fx_kn=np.zeros((mpc.NUM_STAGES,)),
        )
        
        self._input_slew_array = CmdSlewArray.new_with_validation(
            delta_dot_radps=np.zeros((mpc.NUM_STAGES,)),
            fx_dot_knps=np.zeros((mpc.NUM_STAGES,)),
        )


        self.nlp_run.nlp.params.set_value(
            key="stage_cost", value=config.STAGE_OBJ_PARAMS.to_array()
        )
        self.nlp_run.nlp.params.set_value(
            key="term_cost", value=config.TERM_OBJ_PARAMS.to_array()
        )

        self.nlp_run.nlp.states.set_init_guess("ux_mps", ux_guess)
        self.nlp_run.nlp.states.set_init_guess("uy_mps", 0.5)
        self.nlp_run.nlp.states.set_init_guess("r_radps", 0.0)
        self.nlp_run.nlp.states.set_init_guess("dfz_long_kn", 0.1)
        self.nlp_run.nlp.states.set_init_guess("dfz_lat_kn", 0.1)
        self.nlp_run.nlp.states.set_init_guess("s_m", np.linspace(0, 10, mpc.NUM_STAGES))
        self.nlp_run.nlp.states.set_init_guess("e_m", 0.0)
        self.nlp_run.nlp.states.set_init_guess("dpsi_rad", 0.0)
        self.nlp_run.nlp.states.set_init_guess("delta_rad", 0.0)
        self.nlp_run.nlp.states.set_init_guess("fx_kn", 0.0)
        self.nlp_run.update_init_guess()


    def get_states_at_t(self, t: float) -> st._StatesPath:
        """
        Interpolates into the current horizon to get the states at time t.
        """
        interp_states = self._state_array.interp1d(x=t, xp=self._t_array)
        return st._StatesPath.from_array(interp_states)
    
    def get_inputs_at_t(self, t: float) -> st._Inputs:
        """
        Interpolates into the current horizon to get the inputs at time t.
        """
        interp_inputs =  self._input_array.interp1d(x=t, xp=self._t_array)
        return st._Inputs.from_array(interp_inputs)
    
    def get_input_slews_at_t(self, t: float) -> st._Inputs:
        """
        Interpolates into the current horizon to get the input slews at time t.
        """
        interp_input_slews = self._input_slew_array.interp1d(x=t, xp=self._t_array)
        return st._InputSlews.from_array(interp_input_slews)
    

    def pack_nlp_params(self, fromauto_msg: FromAutobox) -> NLPOutput:

        self._nlp_t_start = fromauto_msg.t_s + config.REPLAN_TIME

        proj_states = self.get_states_at_t(t=self._nlp_t_start)
        proj_inputs = self.get_inputs_at_t(t=self._nlp_t_start)

        init_cond = mpc.MPCStates(
            ux_mps=fromauto_msg.ux_mps,
            uy_mps=fromauto_msg.uy_mps,
            r_radps=fromauto_msg.r_radps,
            dfz_long_kn=proj_states.dfz_long_kn,
            dfz_lat_kn=proj_states.dfz_lat_kn,
            s_m=fromauto_msg.s_m,
            e_m=fromauto_msg.e_m,
            dpsi_rad=fromauto_msg.dpsi_rad,
            delta_rad=proj_inputs.delta_rad,
            fx_kn=proj_inputs.fx_kn,
        )



        init_cond_array = init_cond.to_array()
        self.nlp_run.nlp.params.set_value("init_states", init_cond_array)

        # print(init_cond)

        return NLPSetup(
            heartbeat=fromauto_msg.heartbeat,
            init_time=self._nlp_t_start,
            init_states=init_cond_array,
        )

    def solve_mpc(self, fromauto_msg: FromAutobox) -> NLPOutput:
        setup = self.pack_nlp_params(fromauto_msg=fromauto_msg)
        results, stats = self.nlp_run.run_solver()
        nlp_t_horizon = np.linspace(self._nlp_t_start, self._nlp_t_start + mpc.NUM_STAGES * mpc.STEP_SIZE, int(mpc.NUM_STAGES), endpoint=False).squeeze()

        # Only update our "last-valid" horizon if the solution returns successfully
        if(stats.exit_flag==1 or stats.exit_flag==2):
            self._s_array = np.array(results["s_m"]).squeeze()
            self._t_array = nlp_t_horizon

            self._state_array = StateArray.new_with_validation(
                ux_mps=np.array(results["ux_mps"].squeeze()),
                uy_mps=np.array(results["uy_mps"].squeeze()),
                r_radps=np.array(results["r_radps"].squeeze()),
                dfz_long_kn=np.array(results["dfz_long_kn"].squeeze()),
                dfz_lat_kn=np.array(results["dfz_lat_kn"].squeeze()),
                s_m=np.array(results["s_m"].squeeze()),
                e_m=np.array(results["e_m"].squeeze()),
                dpsi_rad=np.array(results["dpsi_rad"].squeeze()),
            )

            self._input_array = CmdArray.new_with_validation(
                delta_rad=np.array(results["delta_rad"].squeeze()),
                fx_kn=np.array(results["fx_kn"].squeeze()),
            )

            self._input_slew_array = CmdSlewArray.new_with_validation(
                delta_dot_radps=np.array(results["delta_dot_radps"].squeeze()),
                fx_dot_knps=np.array(results["fx_dot_knps"].squeeze()),
            )


        result = NLPResult(
            ux_mps=results["ux_mps"].squeeze(),
            uy_mps=results["uy_mps"].squeeze(),
            r_radps=results["r_radps"].squeeze(),
            dfz_long_kn=results["dfz_long_kn"].squeeze(),
            dfz_lat_kn=results["dfz_lat_kn"].squeeze(),
            s_m=results["s_m"].squeeze(),
            e_m=results["e_m"].squeeze(),
            dpsi_rad=results["dpsi_rad"].squeeze(),
            delta_rad=results["delta_rad"].squeeze(),
            fx_kn=results["fx_kn"].squeeze(),
            delta_dot_radps=results["delta_dot_radps"].squeeze(),
            fx_dot_knps=results["fx_dot_knps"].squeeze(),
            t_s=nlp_t_horizon,
            solve_time_s=stats.solve_time_ms/1000.0,
            exit_flag=stats.exit_flag,
            iterations=stats.iterations,
        )

        return NLPOutput(result=result, setup=setup)


class MPCRunnerNode(node.Node):
    def __init__(self, name="mpc_runner") -> None:
        super().__init__(name)
        self.runner = None

    def setup_node(self, nlp_run: nlp_runner.NLPRunner) -> None:

        self.runner = MPCRunner(
            nlp_run=nlp_run,
        )
        self.nlpout_pub = self.create_publisher(
            msg_type=NLPOutput,
            topic="/rooster/nlp_out",
            qos_profile=_PUBSUB_QOS,
        )

        self.nlp_client = self.create_service(
            srv_type=NLPService,
            srv_name="/rooster_srv/nlp",
            qos_profile=_SERV_QOS,
            callback=self.serv_callback,
        )

    def serv_callback(
        self, request: NLPService.Request, response: NLPService.Response
    ) -> NLPService.Response:
        
        start_time = time.perf_counter()
        NLPOutput = self.runner.solve_mpc(fromauto_msg=request.fromauto_msg)

        delta_rad = np.array(NLPOutput.result.delta_rad).squeeze()
        fx_kn = np.array(NLPOutput.result.fx_kn).squeeze()

        response.t_cmd_s = NLPOutput.result.t_s
        response.s_cmd_m = NLPOutput.result.s_m
        response.exit_flag = NLPOutput.result.exit_flag
        response.delta_cmd_rad = delta_rad.tolist()
        response.fx_cmd_kn = fx_kn.tolist()

        NLPOutput.result.total_time = (time.perf_counter() - start_time) * 1000.0
        response.total_time = NLPOutput.result.total_time

        NLPOutput.header.stamp = self.get_clock().now().to_msg()
        self.nlpout_pub.publish(msg=NLPOutput)

        response.header.stamp = self.get_clock().now().to_msg()

        if response.exit_flag==1 or response.exit_flag==2:
            self.get_logger().info(
                f"Solver running normally in {round(response.total_time,1)} ms ({NLPOutput.result.iterations} iterations) at s_m: {round(request.fromauto_msg.s_m,1)}, e_meas: {round(request.fromauto_msg.e_m, 2)}, ux_meas: {round(request.fromauto_msg.ux_mps, 1)}.",
                throttle_duration_sec=0.05,
                throttle_time_source_type=self.get_clock(),
            )
        else:
            self.get_logger().warn(
                f"Solver failed with exit flag {response.exit_flag}",
            )

        return response


def main(args=None):
    """
    Entrypoint for ros2 executable.

    Instantiates a CoordinatorNode object and spins until shutdown.

    """
    rclpy.init(args=args)
    nlp_run = nlp_runner.NLPRunner.load_nlp(
        so_file=SOLVER_PATH, solver_name="ipopt", solver_opts=config.SOLVER_OPTS
    )
    mpc_node = MPCRunnerNode()

    try:
        mpc_node.setup_node(nlp_run=nlp_run)
        rclpy.spin(node=mpc_node)
    except KeyboardInterrupt:
        pass
    finally:
        mpc_node.destroy_node()


if __name__ == "__main__":
    main()
