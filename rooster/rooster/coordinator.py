"""Implement the coordinator node for a rooster simulation or experiment."""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import rclpy
from auto_msgs2.msg import FromAutobox, ToAutobox
from models import world as wd
from rclpy import node, qos
from rooster_msgs.srv import NLPService

from rooster import config, utils

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
class Coordinator:
    """
    Implements coordinator class for synchronizing activities in rooster experiment.
    """

    world: wd.SimpleWorld
    mpc_caller: Callable[[FromAutobox], asyncio.Future]

    _nlp_result: NLPService.Response = field(init=False, repr=False)
    _nlp_future: asyncio.Future[NLPService.Response] = field(init=False, repr=False)
    _num_fromauto_rcv: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._num_fromauto_rcv = 5

        self.start_time = 0.0

        # Initialize the _nlp_result variable with a no-action horizon
        self._nlp_result = NLPService.Response(
            total_time=-1717.0,
            exit_flag=1,
            t_cmd_s=np.array([-1e6, 1e6]),
            s_cmd_m=np.array([-1e6, 1e6]),
            delta_cmd_rad=np.zeros((2,)),
            fx_cmd_kn=np.zeros((2,)),
        )
        self._nlp_future = asyncio.Future()
        self._nlp_future.set_result(self._nlp_result)
        self._last_exit_flag = -1717.0
        self._last_solve_duration_s = -1717.0

    def fromauto_callback(self, msg: FromAutobox, logger) -> ToAutobox:
        """
        Respond to an incoming FromAutobox message.
        """
        
        self._num_fromauto_rcv += 1
        states, _ = utils.get_states_inputs_from_fromautobox(msg)

        # Update the map matched s, e, and dpsi based off the most recent East/North/Psi measurement
        msg.s_m, msg.e_m, msg.dpsi_rad = self.world.enu_to_seu(
            east_m=states.east_m, north_m=states.north_m, psi_rad=states.psi_rad
        )

        # Logic for determining when we update our motion plan, stored in self._nlp_result

        # If we have a new solution and have passed our NUM_FROMAUTO_BEFORE_REPLAN mark,
        # then update our current motion plan
        if (
            self._num_fromauto_rcv >= config.NUM_FROMAUTO_BEFORE_REPLAN
            and self._nlp_future.done()
        ):
            result = deepcopy(self._nlp_future.result())
            self._last_solve_duration_s = time.perf_counter() - self.start_time
            self._nlp_future = self.mpc_caller(msg) # Kick off a new trajectory optimization
            self._num_fromauto_rcv = 1
            self.start_time = time.perf_counter()

            self._last_exit_flag = float(result.exit_flag)

            if result.exit_flag == 1:
                self._nlp_result = result

        # Otherwise, don't do anything---just wait for the self._nlp_future to finish
        else:
            pass

        
        # Interpolate into our current motion plan to get the control inputs
        delta_cmd_rad, fx_cmd_kn = self._interpolate_inputs_s(msg.s_m)

        # Edge case safeties for when all the nodes aren't spun up yet
        if(np.isnan(delta_cmd_rad)): delta_cmd_rad = 0.0
        if(np.isnan(fx_cmd_kn)): fx_cmd_kn = 0.0

        # Finally, package the control commands and other relevant signals in a ToAutobox message
        return ToAutobox(
            heartbeat=msg.heartbeat,
            t_s=msg.t_s,
            s_m=msg.s_m,
            e_m=msg.e_m,
            dpsi_rad=msg.dpsi_rad,
            delta_cmd_rad=delta_cmd_rad,
            fx_cmd_kn=fx_cmd_kn,
            solver_exit_flag=self._last_exit_flag,
            solver_solve_time_s=self._last_solve_duration_s,
            user_def0=0.0,
            user_def1=0.0,
            user_def2=0.0,
            user_def3=0.0,
        )

    def _interpolate_inputs_s(self, s_m: float) -> Tuple[float, float, float]:

        delta_cmd_rad = np.interp(s_m, self._nlp_result.s_cmd_m, self._nlp_result.delta_cmd_rad)
        fx_cmd_kn = np.interp(s_m, self._nlp_result.s_cmd_m, self._nlp_result.fx_cmd_kn)

        return delta_cmd_rad, fx_cmd_kn


class CoordinatorNode(node.Node):
    """
    Implements the main coordinator node for rooster experiments.
    """

    def __init__(self, name="coordinator") -> None:
        super().__init__(name)
        self.coord = None

    def setup_node(self) -> None:
        self.coord = Coordinator(world=config.WORLD, mpc_caller=self.call_mpc)
        self.fromauto_sub = self.create_subscription(
            msg_type=FromAutobox,
            topic="/auto_bridge2/from_autobox",
            callback=self.fromauto_callback,
            qos_profile=_PUBSUB_QOS,
        )

        self.toauto_pub = self.create_publisher(
            msg_type=ToAutobox,
            topic="/auto_bridge2/to_autobox",
            qos_profile=_PUBSUB_QOS,
        )

        self.nlp_client = self.create_client(
            srv_type=NLPService,
            srv_name="/rooster_srv/nlp",
            qos_profile=_SERV_QOS,
        )

        while not self.nlp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("NLP Service not available, waiting again...")

    def fromauto_callback(self, msg: FromAutobox) -> None:
        """
        Callback function for FromAutobox messages.
        """

        toauto_msg = self.coord.fromauto_callback(msg=msg, logger=self.get_logger())

        toauto_msg.header.stamp = self.get_clock().now().to_msg()
        self.toauto_pub.publish(toauto_msg)

    def call_mpc(self, fromauto_msg: FromAutobox) -> asyncio.Future:
        nlp_request = NLPService.Request(fromauto_msg=fromauto_msg)
        nlp_request.header.stamp = self.get_clock().now().to_msg()
        return self.nlp_client.call_async(nlp_request)


def main(args=None):
    """
    Entrypoint for ros2 executable.
    """

    rclpy.init(args=args)
    coord_node = CoordinatorNode()

    try:
        coord_node.setup_node()
        rclpy.spin(node=coord_node)
    except KeyboardInterrupt:
        pass
    finally:
        coord_node.destroy_node()


if __name__ == "__main__":
    main()
