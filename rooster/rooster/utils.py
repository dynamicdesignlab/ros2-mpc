from typing import Tuple

from auto_msgs2.msg import FromAutobox
from models import nn_dynamics as st
#from models import single_track as st


def get_states_inputs_from_fromautobox(
    fromauto_msg: FromAutobox,
) -> Tuple[st._StatesGlobal, st._Inputs, st._StatesPast]:
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
        delta_rad=fromauto_msg.delta_est_rad,
        fx_kn=fromauto_msg.fx_est_kn,
    )
    out_past = st._StatesPast(
    r_2 = fromauto_msg.r_2,
    r_1 = fromauto_msg.r_1,
    uy_2 = fromauto_msg.uy_2,
    uy_1 = fromauto_msg.uy_1,
    ux_2 = fromauto_msg.ux_2,
    ux_1 = fromauto_msg.ux_1,
    delta_2 = fromauto_msg.delta_2,
    delta_1 = fromauto_msg.delta_1,
    fx_2 = fromauto_msg.fx_2,
    fx_1 = fromauto_msg.fx_1,
    )

    return out_state, out_input, out_past
