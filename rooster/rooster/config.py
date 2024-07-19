from pathlib import Path

import numpy as np
from ament_index_python import packages
from casadi_tools.nlp_utils import nlp_problem as nlp
from models import fiala_brush_tire as fb
from models import single_track as st
from models import vehicle_params as vp
from models import world as wd

from rooster import interpolation, mpc_formulation

ROOT_DIR = Path().home().joinpath("Workspace").joinpath("src").joinpath("rooster")
ROOSTER_DIR = ROOT_DIR.joinpath("rooster")
CODEGEN_DIR = ROOSTER_DIR.joinpath("codegen")
PARAM_DIR = Path(packages.get_package_share_directory("rooster")).joinpath("config")

# Load in world map
WORLD_DIR = PARAM_DIR.joinpath("worlds")
WORLD_FILE = WORLD_DIR.joinpath("small_oval_rooster_beta.mat")
WORLD = wd.SimpleWorld.load_from_mat(WORLD_FILE)

# Load in vehicle parameters to be used in motion planning
VEH_DIR = PARAM_DIR.joinpath("vehicles")
VEHICLE_PARAM_FILE = VEH_DIR.joinpath("niki_parameters.yaml")
VEHICLE_PARAMS = vp.VehicleParams.load_from_yaml(VEHICLE_PARAM_FILE)
FRONT_TIRE = fb.FialaBrushTire.load_from_yaml(VEHICLE_PARAM_FILE, "front")
REAR_TIRE = fb.FialaBrushTire.load_from_yaml(VEHICLE_PARAM_FILE, "rear")
VEHICLE_MODEL = st.Model(params=VEHICLE_PARAMS, f_tire=FRONT_TIRE, r_tire=REAR_TIRE)

# Load in vehicle parameters to be used in simulation (for now they're the same as the motion planning ones)
SIM_VEHICLE_PARAMS = vp.VehicleParams.load_from_yaml(VEHICLE_PARAM_FILE)
SIM_FRONT_TIRE = fb.FialaBrushTire.load_from_yaml(VEHICLE_PARAM_FILE, "front")
SIM_REAR_TIRE = fb.FialaBrushTire.load_from_yaml(VEHICLE_PARAM_FILE, "rear")
SIM_VEHICLE_MODEL = st.Model(params=SIM_VEHICLE_PARAMS, f_tire=SIM_FRONT_TIRE, r_tire=SIM_REAR_TIRE)

# Configure NLP settings
MPC_SOLVER_NAME = "rooster_nlp"
MPC_SOLVER_PATH = CODEGEN_DIR.joinpath(MPC_SOLVER_NAME).with_suffix(".so")
MPC_GEN_OPT_LEVEL = 3
IPOPT_OPTS = {
    "sb": "yes",
    "print_level": 1,
    "max_iter": 1000,
    "linear_solver": "ma57",
}
SOLVER_OPTS = {"ipopt": IPOPT_OPTS, "print_time": 0}

# Set the weights of the stagewise MPC cost function
STAGE_OBJ_PARAMS = mpc_formulation.StageObjParams(
    w_ux        = 1.0 / (2.0**2),
    w_e         = 1.0 / (0.2**2),
    w_dpsi      = 1.0 / (np.radians(10)**2),
    w_delta_dot = 1.0 / (np.radians(10)**2),
    w_fx_dot    = 1.0 / (1.0**2),
)

# Set the weights of the terminal cost MPC cost function
TERM_OBJ_PARAMS = mpc_formulation.TerminalObjParams(
    w_terminal_ux        = 1.0 / (2.0**2),
    w_terminal_e         = 1.0 / (0.2**2),
    w_terminal_dpsi      = 1.0 / (np.radians(10)**2),
)

# Configure additional properties of the MPC problem
MPC_PROBLEM = mpc_formulation.MPCFormulation(
    model=VEHICLE_MODEL,
    interp_ux=interpolation.get_interp_ux(world=WORLD, double=True),
    interp_k_psi=interpolation.get_interp_k_psi(world=WORLD, double=True),
)

REPLAN_TIME   = 0.05 # seconds
FROMAUTO_TIME = 0.01 # seconds

NUM_FROMAUTO_BEFORE_REPLAN = int(REPLAN_TIME / FROMAUTO_TIME)
