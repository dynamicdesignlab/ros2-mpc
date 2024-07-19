from pathlib import Path

from live_mpl import LiveLine, Tab, Window


from rooster import config
from rooster_viz import plot_dialogs, ros_plot_utils

ROOSTER_VIZ_DIR = Path(__file__).parent.parent
ROSBAG_DIR = ROOSTER_VIZ_DIR.parent.joinpath("rosbags")



def plot_global_position(win: Window, from_data, mpc_data):
    glob_tab = Tab("Global Position")
    win.register_tab(glob_tab)
    glob_axis = glob_tab.add_subplot(
        1, 1, 1,
        ylabel="North Position [m]",
        xlabel="East Position [m]",
        title="Global Vehicle Position",
    )
    glob_axis.plot(config.WORLD.inner_bound_east_m, config.WORLD.inner_bound_north_m, color="gray", label="Track boundary")
    glob_axis.plot(config.WORLD.outer_bound_east_m, config.WORLD.outer_bound_north_m, color="gray")
    glob_axis.plot(from_data["east_m"], from_data["north_m"], label="Vehicle Position")
    plot_glob = LiveLine(
        ax=glob_axis,
        x_data=mpc_data["east_m"],
        y_data=mpc_data["north_m"],
    )
    glob_tab.register_plot(plot_glob)
    glob_axis.axis("equal")
    glob_axis.legend()




def plot_states(win: Window, from_data, mpc_data):
    tab1 = Tab(tab_name="Vehicle states")
    win.register_tab(tab1)


    axis_ux = tab1.add_subplot(
        4, 2, 1,
        xlabel="Time [s]",
        ylabel="u_x [m/s]",
        title="Longitudinal velocity",
    )
    axis_ux.plot(from_data["t_s"], from_data["ux_mps"])
    plot_ux = LiveLine(
        ax=axis_ux,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["ux_mps"],
    )
    tab1.register_plot(plot_ux)

    axis_uy = tab1.add_subplot(
        4, 2, 2,
        xlabel="Time [s]",
        ylabel="u_y [m/s]",
        title="Lateral velocity",
        sharex=axis_ux,
    )
    axis_uy.plot(from_data["t_s"], from_data["uy_mps"])
    plot_uy = LiveLine(
        ax=axis_uy,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["uy_mps"],
    )
    tab1.register_plot(plot_uy)


    axis_r = tab1.add_subplot(
        4, 2, 3,
        xlabel="Time [s]",
        ylabel="r [rad/s]",
        title="Yaw rate",
        sharex=axis_ux,
    )
    axis_r.plot(from_data["t_s"], from_data["r_radps"])
    plot_r = LiveLine(
        ax=axis_r,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["r_radps"],
    )
    tab1.register_plot(plot_r)



    # Map match to obtain s, e, and dpsi
    from_data["s_m"], from_data["e_m"], from_data["dpsi_rad"] = config.WORLD.enu_to_seu(from_data["east_m"], from_data["north_m"], from_data["psi_rad"])

    axis_s = tab1.add_subplot(
        4, 2, 4, xlabel="Time [s]", ylabel="s [m]", title="Centerline progress", sharex=axis_ux,
    )
    axis_s.plot(from_data["t_s"], from_data["s_m"])
    plot_s = LiveLine(
        ax=axis_s,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["s_m"],
    )
    tab1.register_plot(plot_s)

    axis_e = tab1.add_subplot(
        4, 2, 5, xlabel="Time [s]", ylabel="e [m]", title="Lateral position", sharex=axis_ux,
    )
    axis_e.plot(from_data["t_s"], from_data["e_m"])
    plot_e = LiveLine(
        ax=axis_e,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["e_m"],
    )
    tab1.register_plot(plot_e)

    axis_dpsi = tab1.add_subplot(
        4, 2, 6, xlabel="Time [s]", ylabel="dpsi_rad [rad]", title="Delta Psi", sharex=axis_ux,
    )
    axis_dpsi.plot(from_data["t_s"], from_data["dpsi_rad"])
    plot_dpsi = LiveLine(
        ax=axis_dpsi,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dpsi_rad"],
    )
    tab1.register_plot(plot_dpsi)



    axis_dfz_long = tab1.add_subplot(
        4, 2, 7, xlabel="Time [s]", ylabel="dfz_long [kn]", title="Long. load transfer", sharex=axis_ux,
    )
    axis_dfz_long.plot(from_data["t_s"], from_data["dfz_long_est_kn"])
    plot_dfz_long = LiveLine(
        ax=axis_dfz_long,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dfz_long_kn"],
    )
    tab1.register_plot(plot_dfz_long)

    axis_dfz_lat = tab1.add_subplot(
        4, 2, 8, xlabel="Time [s]", ylabel="dfz_lat [kn]", title="Lat. load transfer", sharex=axis_ux,
    )
    axis_dfz_lat.plot(from_data["t_s"], from_data["dfz_lat_est_kn"])
    plot_dfz_lat = LiveLine(
        ax=axis_dfz_lat,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dfz_lat_kn"],
    )
    tab1.register_plot(plot_dfz_lat)

    



def plot_inputs(win: Window, from_data, mpc_data):
    tab1 = Tab(tab_name="Vehicle inputs")
    win.register_tab(tab1)


    axis_delta = tab1.add_subplot(
        2, 1, 1, xlabel="Time [s]", ylabel="delta [rad]", title="Steering angle",
    )
    axis_delta.plot(from_data["t_s"], from_data["delta_cmd_rad"], label="Command")
    axis_delta.plot(from_data["t_s"], from_data["delta_meas_rad"], label="Measured")
    plot_delta = LiveLine(
        ax=axis_delta,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["delta_rad"],
    )
    axis_delta.legend(loc="upper right")
    tab1.register_plot(plot_delta)


    axis_fx = tab1.add_subplot(
        2, 1, 2, xlabel="Time [s]", ylabel="Fx [kn]", title="Longitudinal force", sharex=axis_delta,
    )
    axis_fx.plot(from_data["t_s"], from_data["fx_cmd_kn"], label="Command")
    axis_fx.plot(from_data["t_s"], from_data["fx_meas_kn"], label="Measured")
    plot_fx = LiveLine(
        ax=axis_fx,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["fx_kn"],
    )
    axis_delta.legend(loc="upper right")
    tab1.register_plot(plot_fx)




def plot_solver_stats(win: Window, mpc_data):
    tab1 = Tab(tab_name="Solver Stats")
    win.register_tab(tab1)

    axis_time = tab1.add_subplot(
        3, 1, 1,
        xlabel="Time [s]",
        ylabel="Solve time [ms]",
        title="Solver Time"
    )
    axis_time.step(mpc_data["t0_s"], mpc_data["solve_time_s"]*1000)

    axis_iter = tab1.add_subplot(
        3, 1, 2,
        xlabel="Time [s]",
        ylabel="Iterations [-]",
        title="Solver Iterations"
    )
    axis_iter.step(mpc_data["t0_s"], mpc_data["iterations"])

    axis_exit = tab1.add_subplot(
        3, 1, 3,
        xlabel="Time [s]",
        ylabel="Exit flag [-]",
        title="Solver exit flag"
    )
    axis_exit.step(mpc_data["t0_s"], mpc_data["exit_flag"])
    
    

def main():
    data_path = plot_dialogs.select_file_dialog(
        ROSBAG_DIR, label="Rosbag Files", filt_pattern="*.db3"
    )
    data = ros_plot_utils.parse_rosbag_for_plotting(data_path)

    win = Window("rooster_plot")

    plot_global_position(win, data.from_data, data.mpc_data)
    plot_states(win, data.from_data, data.mpc_data)
    plot_inputs(win, data.from_data, data.mpc_data)
    plot_solver_stats(win, data.mpc_data)

    win.loop()


if __name__ == "__main__":
    main()
