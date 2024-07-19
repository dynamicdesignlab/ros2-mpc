# ---------------- NLPResult.msg ------------------- #

std_msgs/Header header

# MPC States
float64[] ux_mps
float64[] uy_mps
float64[] r_radps
float64[] dfz_long_kn
float64[] dfz_lat_kn
float64[] s_m
float64[] e_m
float64[] dpsi_rad
float64[] delta_rad
float64[] fx_kn

# MPC Inputs
float64[] delta_dot_radps
float64[] fx_dot_knps

# MPC Misc
float64[] t_s

# Solver Stats
float64 solve_time_s
int64 exit_flag
int64 iterations
float64 total_time