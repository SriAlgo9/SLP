import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Velocity validation at two times
# ==========================================================

def plot_velocity_validation(
    x0,
    uy0,
    x1,
    uy1,
    x_points_t0,
    uy_points_t0,
    x_points_t1,
    uy_points_t1,
):

    plt.figure(figsize=(8,5))

    # Numerical t = 0
    plt.plot(
        x0,
        uy0,
        color="blue",
        linewidth=2,
        label="Numerical t = 0 s"
    )

    # Analytical t = 0 (dots)
    plt.scatter(
        x_points_t0,
        uy_points_t0,
        color="blue",
        s=40,
        label="Analytical t = 0 s"
    )

    # Numerical t = 1
    plt.plot(
        x1,
        uy1,
        color="red",
        linewidth=2,
        label="Numerical t = 1 s"
    )

    # Analytical t = 1 (dots)
    plt.scatter(
        x_points_t1,
        uy_points_t1,
        color="red",
        s=40,
        label="Analytical t = 1 s"
    )

    plt.xlabel("x")
    plt.ylabel(r"$u_y$")
    plt.title("Horizontal Centreline Velocity Validation")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
