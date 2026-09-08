import numpy as np

from scipy.sparse.linalg import factorized

from src.utils.boundary_conditions import (
    velocity_bc,
)

from src.utils.velocity_correction import (
    compute_ux,
    compute_uy,
)

from src.utils.explicit_scheme import (
    compute_ux_star,
    compute_uy_star,
)

from src.utils.diagnostics import (
    compute_velocity_magnitude,
    compute_divergence,
)

from src.solvers.poisson_matrix import (
    poisson_matrix,
    build_rhs,
)


def run_simulation(
    ux,
    uy,
    p,
    rho,
    nu,
    dx,
    dy,
    dt,
    nt,
    Nx,
    Ny,
):
    """
    Run the Navier-Stokes simulation.
    """

    A = poisson_matrix(Nx, Ny, dx)

    A[0, :] = 0.0
    A[0, 0] = 1.0

    A = A.tocsc()

    solve_poisson = factorized(A)

    ux_history = [ux.copy()]
    uy_history = [uy.copy()]
    p_history = [p.copy()]
    time_history = [0.0]

    for n in range(nt):

        ux_star = compute_ux_star(
            ux,
            uy,
            dt,
            dx,
            dy,
            nu
        )

        uy_star = compute_uy_star(
            ux,
            uy,
            dt,
            dx,
            dy,
            nu
        )

        ux_star, uy_star = velocity_bc(
            ux_star,
            uy_star
        )

        b = build_rhs(
            ux_star,
            uy_star,
            rho,
            dt,
            dx,
            dy
        )

        b[0] = 0.0

        p_vec = solve_poisson(b)

        p = p_vec.reshape((Ny, Nx))

        # p = pressure_bc(p)

        ux = compute_ux(
            ux_star,
            dt,
            dx,
            rho,
            p
        )

        uy = compute_uy(
            uy_star,
            dt,
            dy,
            rho,
            p
        )

        ux, uy = velocity_bc(
            ux,
            uy
        )

        _, _, speed = compute_velocity_magnitude(
            ux,
            uy
        )

        div = compute_divergence(
            ux,
            uy,
            dx,
            dy
        )

        ux_history.append(ux.copy())
        uy_history.append(uy.copy())
        p_history.append(p.copy())
        time_history.append((n + 1) * dt)

        if n % 500 == 0:

            print(
                f"Step {n:5d} | "
                f"max speed = {np.max(speed):.6f} | "
                f"max divergence = {np.max(np.abs(div)):.6e}"
            )

    return (
        ux,
        uy,
        p,
        ux_history,
        uy_history,
        p_history,
        time_history,
    )