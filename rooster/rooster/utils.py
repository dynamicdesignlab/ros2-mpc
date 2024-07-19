from typing import Tuple

from auto_msgs2.msg import FromAutobox
from models import single_track as st


def get_states_inputs_from_fromautobox(
    fromauto_msg: FromAutobox,
) -> Tuple[st._StatesGlobal, st._Inputs]:
    """
    Create NamedVector state and inputs instances from a FromAutobox message.
    """
    
    out_state = st._StatesGlobal(
        ux_mps=fromauto_msg.ux_mps,
        uy_mps=fromauto_msg.uy_mps,
        r_radps=fromauto_msg.r_radps,
        dfz_long_kn=0.0,
        dfz_lat_kn=0.0,
        east_m=fromauto_msg.east_m,
        north_m=fromauto_msg.north_m,
        psi_rad=fromauto_msg.psi_rad,
    )
    out_input = st._Inputs(
        delta_rad=fromauto_msg.delta_meas_rad,
        fx_kn=fromauto_msg.fx_meas_kn,
    )

    return out_state, out_input
