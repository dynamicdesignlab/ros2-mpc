from pathlib import Path

from live_mpl import LiveLine, Tab, Window


import numpy as np
from rooster import config
from rooster_viz import plot_dialogs, ros_plot_utils
import scipy.interpolate

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

    e_ref_interpolant = scipy.interpolate.interp1d(config.WORLD.s_m, config.WORLD.e_ref_m)
    east_ref, north_ref, _, _ = config.WORLD.sebdpsi_to_enupsi(from_data["s_m"], e_ref_interpolant(from_data["s_m"] % config.WORLD.length_m), np.zeros_like(from_data["s_m"]), np.zeros_like(from_data["s_m"]))
    glob_axis.plot(east_ref, north_ref, ls="--", color="k", label="Reference")
    glob_axis.plot(from_data["east_m"], from_data["north_m"], color="black", label="Measured position")
    plot_glob = LiveLine(
        ax=glob_axis,
        x_data=mpc_data["east_m"],
        y_data=mpc_data["north_m"],
        plot_kwargs={"marker":"o", "color":"royalblue", "markeredgewidth": 2, "fillstyle":"none"}
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
        ylabel=r"$u_x$ [m/s]",
    )
    axis_ux.plot(from_data["t_s"], from_data["ux_mps"], color="black")
    ux_ref_interpolant = scipy.interpolate.interp1d(config.WORLD.s_m, config.WORLD.ux_ref_mps)
    axis_ux.plot(from_data["t_s"], ux_ref_interpolant(from_data["s_m"]), ls="--", color="k", label="Reference")
    plot_ux = LiveLine(
        ax=axis_ux,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["ux_mps"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_ux)

    axis_uy = tab1.add_subplot(
        4, 2, 2,
        xlabel="Time [s]",
        ylabel=r"$u_y$ [m/s]",
        sharex=axis_ux,
    )
    axis_uy.plot(from_data["t_s"], from_data["uy_mps"], color="black")
    plot_uy = LiveLine(
        ax=axis_uy,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["uy_mps"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_uy)


    axis_r = tab1.add_subplot(
        4, 2, 3,
        xlabel="Time [s]",
        ylabel=r"$r$ [rad/s]",
        sharex=axis_ux,
    )
    axis_r.plot(from_data["t_s"], from_data["r_radps"], color="black")
    plot_r = LiveLine(
        ax=axis_r,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["r_radps"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_r)



    # Map match to obtain s, e, and dpsi
    from_data["s_m"], from_data["e_m"], _, from_data["dpsi_rad"] = config.WORLD.enupsi_to_sebdpsi(from_data["east_m"], from_data["north_m"], from_data["up_m"], from_data["psi_rad"])

    axis_s = tab1.add_subplot(
        4, 2, 4, xlabel="Time [s]", ylabel="s [m]", title="Centerline progress", sharex=axis_ux,
    )
    axis_s.plot(from_data["t_s"], from_data["s_m"], color="black")
    plot_s = LiveLine(
        ax=axis_s,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["s_m"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_s)

    axis_e = tab1.add_subplot(
        4, 2, 5, xlabel="Time [s]", ylabel=r"$e$ [m]", sharex=axis_ux,
    )
    axis_e.plot(from_data["t_s"], from_data["e_m"], color="black")
    e_ref_interpolant = scipy.interpolate.interp1d(config.WORLD.s_m, config.WORLD.e_ref_m)
    axis_e.plot(from_data["t_s"], e_ref_interpolant(from_data["s_m"]), ls="--", color="k", label="Reference")
    plot_e = LiveLine(
        ax=axis_e,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["e_m"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_e)

    axis_dpsi = tab1.add_subplot(
        4, 2, 6, xlabel="Time [s]", ylabel=r"$\Delta\Psi$ [rad]", sharex=axis_ux,
    )
    axis_dpsi.plot(from_data["t_s"], from_data["dpsi_rad"], color="black")
    plot_dpsi = LiveLine(
        ax=axis_dpsi,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dpsi_rad"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_dpsi)



    axis_dfz_long = tab1.add_subplot(
        4, 2, 7, xlabel="Time [s]", ylabel=r"$\Delta F_{z,long}$ [kn]", sharex=axis_ux,
    )
    axis_dfz_long.plot(from_data["t_s"], from_data["dfz_long_est_kn"], color="black")
    plot_dfz_long = LiveLine(
        ax=axis_dfz_long,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dfz_long_kn"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_dfz_long)

    axis_dfz_lat = tab1.add_subplot(
        4, 2, 8, xlabel="Time [s]", ylabel=r"$\Delta F_{z, lat}$ [kn]", title="Lat. load transfer", sharex=axis_ux,
    )
    axis_dfz_lat.plot(from_data["t_s"], from_data["dfz_lat_est_kn"], color="black")
    plot_dfz_lat = LiveLine(
        ax=axis_dfz_lat,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["dfz_lat_kn"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    tab1.register_plot(plot_dfz_lat)

    



def plot_inputs(win: Window, from_data, mpc_data):
    tab1 = Tab(tab_name="Vehicle inputs")
    win.register_tab(tab1)


    axis_delta = tab1.add_subplot(
        2, 1, 1, xlabel="Time [s]", ylabel="delta [rad]", title="Steering angle",
    )
    axis_delta.plot(from_data["t_s"], from_data["delta_cmd_rad"], label="Command", ls=":", color="black")
    axis_delta.plot(from_data["t_s"], from_data["delta_est_rad"], label="Measured", color="black")
    plot_delta = LiveLine(
        ax=axis_delta,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["delta_rad"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
    )
    axis_delta.legend(loc="upper right")
    tab1.register_plot(plot_delta)


    axis_fx = tab1.add_subplot(
        2, 1, 2, xlabel="Time [s]", ylabel="Fx [kn]", title="Longitudinal force", sharex=axis_delta,
    )
    axis_fx.plot(from_data["t_s"], from_data["fx_cmd_kn"], label="Command")
    axis_fx.plot(from_data["t_s"], from_data["fx_est_kn"], label="Measured")
    plot_fx = LiveLine(
        ax=axis_fx,
        x_data=mpc_data["t_s"],
        y_data=mpc_data["fx_kn"],
        plot_kwargs={"marker":"o", "color":"royalblue", "fillstyle":"none"}
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
