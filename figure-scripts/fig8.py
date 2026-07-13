import os

import matplotlib.pyplot as plt
import numpy as np


#
# Figure 8
# All-electron orbital versus pseudo-orbital
#


def main():
    r = np.linspace(0.0, 5.0, 600)
    r_core = 1.20

    all_electron = all_electron_model(r)
    pseudo = pseudo_orbital_model(r, r_core)

    fig, ax = plt.subplots(1, 1, dpi=144, figsize=(7, 3.2))
    lc = "#00b9f2"

    ax.axvspan(0, r_core, color=lc, alpha=0.18, label=r"core region")
    ax.plot(r, all_electron, color="black", linewidth=1.2,
            linestyle="--",
            label="all-electron valence orbital")
    ax.plot(r, pseudo, color=lc, linewidth=1.8,
            label="pseudo-valence orbital")
    ax.axvline(r_core, color="black", linestyle="--", linewidth=0.8)
    ax.text(r_core + 0.06, 0.93 * ax.get_ylim()[1], r"$r_{\mathrm{c}}$",
            ha="left", va="top", fontsize=9)

    ax.set_xlabel(r"distance from nucleus $r$ [a.u.]")
    ax.set_ylabel("orbital amplitude [arb. units]")
    ax.set_title("Pseudo-orbital smoothing inside the core region")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    output = os.path.join(os.path.dirname(__file__), "..", "img",
                          "fig8_pseudo_orbital.pdf")
    plt.tight_layout()
    plt.savefig(output)


def all_electron_model(r):
    """
    Representative valence radial orbital with a core-region node.

    The Gaussian depression mimics the rapid near-core structure caused by
    orthogonality to lower-lying core states. It is pedagogical, not fitted.
    """
    zeta = 0.55
    amp = 1.85
    sigma = 0.43
    return (1.0 - amp * np.exp(-(r / sigma)**2)) * np.exp(-zeta * r)


def all_electron_model_derivative(r):
    zeta = 0.55
    amp = 1.85
    sigma = 0.43
    gaussian = np.exp(-(r / sigma)**2)
    return np.exp(-zeta * r) * (
        amp * gaussian * 2.0 * r / sigma**2
        - zeta * (1.0 - amp * gaussian)
    )


def pseudo_orbital_model(r, r_core):
    """
    Smooth nodeless pseudo-orbital inside r_core.

    For r < r_core, use c0 + c2 r^2 + c4 r^4.  The even polynomial has zero
    slope at the origin and is matched in value and slope to the all-electron
    model at r_core.  For r >= r_core the two curves are identical.
    """
    pseudo = all_electron_model(r)
    value_core = all_electron_model(r_core)
    slope_core = all_electron_model_derivative(r_core)

    c0 = 0.70 * value_core
    matrix = np.array([[r_core**2, r_core**4],
                       [2.0 * r_core, 4.0 * r_core**3]])
    rhs = np.array([value_core - c0, slope_core])
    c2, c4 = np.linalg.solve(matrix, rhs)

    inside = r < r_core
    pseudo[inside] = c0 + c2 * r[inside]**2 + c4 * r[inside]**4
    return pseudo


if __name__ == "__main__":
    main()
