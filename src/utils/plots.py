import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Velocity validation at multiple times
# ==========================================================

def plot_velocity_validation(
    x_num_all,
    uy_num_all,
    x_exact_all,
    uy_exact_all,
    times,
    dx ,
    CFL,
    rc
):

    plt.figure(figsize=(9, 6))

    colors = plt.cm.viridis(
        np.linspace(0.0, 1.0, max(len(times), 1))
    )

    for color, t, x_num, uy_num, x_exact, uy_exact in zip(
        colors,
        times,
        x_num_all,
        uy_num_all,
        x_exact_all,
        uy_exact_all,
    ):
        plt.plot(
            x_num,
            uy_num,
            color=color,
            linewidth=2,
            label=f"Numerical ({t:.2f} s)"
        )

        plt.plot(
            x_exact,
            uy_exact,
            "--o",
            color=color,
            linewidth=1.5,
            markersize=3,
            markerfacecolor="white",
            label=f"Analytical ({t:.2f} s)"
        )

    plt.xlabel("x")
    plt.ylabel(r"$u_y$")

    plt.title(
            rf"Horizontal Centreline Velocity Validation "
            rf"($\Delta h$ = {dx:g}, CFL = {CFL:g}, $r_c$ = {rc:g})"
    )

    #plt.title("Horizontal Centreline Velocity Validation")

    plt.grid(True)
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


# ==========================================================
# Vorticity validation
# ==========================================================

def plot_vorticity_validation(
    x_num_all,
    omega_num_all,
    x_exact_all,
    omega_exact_all,
    times,
    dx ,
    CFL,
    rc
):

    colors = plt.cm.viridis(
        np.linspace(0.0, 1.0, max(len(times), 1))
    )

    plt.figure(figsize=(9,6))

    for color, t, x_num, omega_num, x_exact, omega_exact in zip(
        colors,
        times,
        x_num_all,
        omega_num_all,
        x_exact_all,
        omega_exact_all,
    ):

        plt.plot(
            x_num,
            omega_num,
            "-",
            color=color,
            linewidth=2.5,
            label=f"Numerical ((t={t:.2f} s)"
        )

        plt.plot(
            x_exact,
            omega_exact,
            "--o",
            color=color,
            linewidth=2,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"Analytical (t={t:.2f} s)"
        )

    plt.xlabel("x")
    plt.ylabel(r"$\omega$")

    plt.title(
        rf"Horizontal Centreline Vorticity Validation "
        rf"($\Delta h$ = {dx:g}, CFL = {CFL:g}, $r_c$ = {rc:g})"
    )
    #plt.title(r'Horizontal Centreline Vorticity Validation ($\Delta h = 0.003125$, CFL = 0.1, rc = 0.025)')
    #plt.title("Horizontal Centreline Vorticity Validation")
    plt.grid(True)
    plt.legend(
        ncol=2,
        fontsize=8,
        loc="upper right"
    )
    plt.tight_layout()
    plt.show()



# ==========================================================
# Global vs Interior Error
# ==========================================================

def plot_global_vs_interior_error(
    times,
    global_velocity_errors,
    interior_velocity_errors,
    global_vorticity_errors,
    interior_vorticity_errors,
    dx,
    CFL,
    rc
):

    times = np.asarray(times)

    # Convert relative errors to percentages
    global_velocity = np.asarray(global_velocity_errors) * 100
    interior_velocity = np.asarray(interior_velocity_errors) * 100

    global_vorticity = np.asarray(global_vorticity_errors) * 100
    interior_vorticity = np.asarray(interior_vorticity_errors) * 100

    plt.figure(figsize=(9, 6))

    plt.plot(
        times,
        global_velocity,
        "-o",
        linewidth=2,
        label="Global velocity"
    )

    plt.plot(
        times,
        interior_velocity,
        "-o",
        linewidth=2,
        label="Interior velocity"
    )

    plt.plot(
        times,
        global_vorticity,
        "-s",
        linewidth=2,
        label="Global vorticity"
    )

    plt.plot(
        times,
        interior_vorticity,
        "-s",
        linewidth=2,
        label="Interior vorticity"
    )

    plt.xlabel("Time (s)")
    plt.ylabel(r"Relative $L_2$ error (%)")

    plt.title(
        rf"Global and Interior Relative $L_2$ Errors "
        rf"($\Delta h$ = {dx:g}, CFL = {CFL:g}, $r_c$ = {rc:g})"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "Global_vs_Interior_Error.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
