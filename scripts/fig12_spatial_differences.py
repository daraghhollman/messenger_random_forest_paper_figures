import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, plotting, utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

wong_colours = {
    "black": "black",
    "orange": "#E69F00",
    "light blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
}


def main():

    # Load full mission data
    full_mission = mag.Load_Mission("./resources/messenger_mag")

    bow_shock_intervals_spread, magnetopause_intervals_spread = get_intervals_spread(
        full_mission
    )
    bow_shock_individual_spread, magnetopause_individual_spread = (
        get_individual_crossing_spread(full_mission)
    )

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)

    # AX = Intervals distributions
    # BX = Individual crossings distributions
    # CX = BX - AX
    # 1 = bow shock
    # 2 = magnetopause
    ((a1, a2), (b1, b2), (c1, c2)) = axes.T

    a1_mesh = plot_density(bow_shock_intervals_spread, a1, label_y=True)
    a2_mesh = plot_density(
        magnetopause_intervals_spread, a2, label_x=True, label_y=True
    )

    b1_mesh = plot_density(bow_shock_individual_spread, b1)
    b2_mesh = plot_density(magnetopause_individual_spread, b2, label_x=True)

    c1_mesh = plot_difference(
        bow_shock_individual_spread, bow_shock_intervals_spread, c1
    )
    c2_mesh = plot_difference(
        magnetopause_individual_spread, magnetopause_intervals_spread, c2, label_x=True
    )

    meshes = [a1_mesh, a2_mesh, b1_mesh, b2_mesh, c1_mesh, c2_mesh]
    cbar_labels = (
        ["Interval Density"] * 2
        + ["Indiv. Crossing Density"] * 2
        + ["B1 - A1"]
        + ["B2 - A2"]
    )
    axis_labels = ["(A1)", "(A2)", "(B1)", "(B2)", "(C1)", "(C2)"]

    for i in range(len(meshes)):
        ax = axes.T.flatten()[i]

        ax.text(4, 7, axis_labels[i], ha="center", fontsize="large").set_clip_on(False)
        ax.set_ylim(0, 8)

        im = meshes[i]

        # Create a divider for existing axes
        divider = make_axes_locatable(ax)

        # Append a new axes to the right of the image, for the colorbar
        cax = divider.append_axes("right", size="5%", pad=0)

        # Add colorbar in the new axes
        fig.colorbar(im, cax=cax, orientation="vertical", label=cbar_labels[i])

    b1.set_title("Bow Shock")
    b2.set_title("Magnetopause")

    # plt.show()
    plt.tight_layout()
    plt.savefig(
        "./figures/fig12_spatial_difference.pdf",
        format="pdf",
    )


def plot_density(hist, ax, label_x=False, label_y=False):
    bin_size = 0.5
    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 10 + bin_size, bin_size)

    mesh = ax.pcolormesh(x_bins, cyl_bins, hist.T, norm="log")

    if label_x:
        ax.set_xlabel(r"$X_{\text{MSM'}} \quad \left[ \text{R}_\text{M} \right]$")
    if label_y:
        ax.set_ylabel(
            r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
        )

    plotting.Plot_Circle(
        ax,
        (0, +utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )
    plotting.Plot_Circle(
        ax,
        (0, -utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )
    plotting.Plot_Magnetospheric_Boundaries(ax, lw=2, zorder=5)
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(cyl_bins[0], cyl_bins[-1])
    ax.set_aspect("equal")

    return mesh


def plot_difference(hist_a, hist_b, ax, label_x=False, label_y=False):
    # Plots a - b
    bin_size = 0.5
    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 10 + bin_size, bin_size)

    cbar_lims = np.nanmax(np.abs(hist_a - hist_b))
    diverging_norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-cbar_lims, vcenter=0, vmax=cbar_lims
    )
    mesh = ax.pcolormesh(
        x_bins, cyl_bins, hist_a.T - hist_b.T, cmap="bwr", norm=diverging_norm
    )

    if label_x:
        ax.set_xlabel(r"$X_{\text{MSM'}} \quad \left[ \text{R}_\text{M} \right]$")
    if label_y:
        ax.set_ylabel(
            r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
        )

    plotting.Plot_Circle(
        ax,
        (0, +utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )
    plotting.Plot_Circle(
        ax,
        (0, -utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )
    plotting.Plot_Magnetospheric_Boundaries(ax, lw=2, zorder=5)
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_ylim(cyl_bins[0], cyl_bins[-1])
    ax.set_aspect("equal")

    return mesh


def get_individual_crossing_spread(full_mission):

    # Load crossings
    crossings = pd.read_csv("./resources/hollman_2025_crossing_list.csv")
    crossings["Time"] = pd.to_datetime(crossings["Times"])

    # Find the position of each crossing

    # Add on the columns of full_mission for the rows in crossings
    # Does this using the nearest element
    # Its so fast omg
    crossings = pd.merge_asof(
        crossings, full_mission, left_on="Time", right_on="date", direction="nearest"
    )

    bow_shock_crossings = crossings.loc[crossings["Label"].str.contains("BS")].copy()
    magnetopause_crossings = crossings.loc[crossings["Label"].str.contains("MP")].copy()

    # To normalise these distributions by residence, we need the ammount of time spent in each bin.
    # Load full mission to get trajectory
    positions = [
        full_mission["X MSM' (radii)"],
        full_mission["Y MSM' (radii)"],
        full_mission["Z MSM' (radii)"],
    ]

    bin_size = 0.5
    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 10 + bin_size, bin_size)

    # Get residence histograms. These are the frequency of data points. We have
    # loaded 1 second average data.
    residence_cyl, _, _ = np.histogram2d(
        positions[0],
        np.sqrt(positions[1] ** 2 + positions[2] ** 2),
        bins=[x_bins, cyl_bins],
    )

    hist_data = []
    for i, positions in enumerate(
        [bow_shock_crossings, magnetopause_crossings],
    ):

        cyl_hist_data, _, _ = np.histogram2d(
            positions["X MSM' (radii)"],
            np.sqrt(
                positions["Y MSM' (radii)"] ** 2 + positions["Z MSM' (radii)"] ** 2
            ),
            bins=[x_bins, cyl_bins],
        )

        # Normalise
        # Yielding crossings per second
        cyl_hist_data = np.where(
            residence_cyl != 0, cyl_hist_data / residence_cyl, np.nan
        )

        # Multiply by 3600 to get crossings per hour
        cyl_hist_data *= 3600

        # Normalise histogram to sum to 1
        cyl_hist_data /= np.nansum(cyl_hist_data)

        hist_data.append(cyl_hist_data)

    return hist_data


def get_intervals_spread(full_mission):

    # Load crossing intervals
    crossing_intervals = boundaries.Load_Crossings(
        utils.User.CROSSING_LISTS["Philpott"]
    )

    # We need to consider one point for each crossing in this plot
    # We generate a new column with the position of MESSENGER at the
    # time in the middle of the crossing
    crossing_intervals["Mid Time"] = (
        crossing_intervals["Start Time"]
        + (crossing_intervals["End Time"] - crossing_intervals["Start Time"]) / 2
    )

    crossing_intervals = pd.merge_asof(
        crossing_intervals,
        full_mission,
        left_on="Mid Time",
        right_on="date",
        direction="nearest",
    )

    bow_shock_intervals = crossing_intervals.loc[
        crossing_intervals["Type"].str.contains("BS")
    ]
    magnetopause_intervals = crossing_intervals.loc[
        crossing_intervals["Type"].str.contains("MP")
    ]

    bow_shock_locations = bow_shock_intervals[
        ["X MSM' (radii)", "Y MSM' (radii)", "Z MSM' (radii)"]
    ].to_numpy()
    magnetopause_locations = magnetopause_intervals[
        ["X MSM' (radii)", "Y MSM' (radii)", "Z MSM' (radii)"]
    ].to_numpy()

    # To normalise these distributions by residence, we need the ammount of time spent in each bin.
    # Load full mission to get trajectory
    positions = [
        full_mission["X MSM' (radii)"],
        full_mission["Y MSM' (radii)"],
        full_mission["Z MSM' (radii)"],
    ]

    bin_size = 0.5
    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    cyl_bins = np.arange(0, 10 + bin_size, bin_size)

    # Get residence histograms. These are the frequency of data points. We have
    # loaded 1 second average data.
    residence_cyl, _, _ = np.histogram2d(
        positions[0],
        np.sqrt(positions[1] ** 2 + positions[2] ** 2),
        bins=[x_bins, cyl_bins],
    )

    hist_data = []
    for positions in [bow_shock_locations, magnetopause_locations]:

        cyl_hist_data, _, _ = np.histogram2d(
            positions[:, 0],
            np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2),
            bins=[x_bins, cyl_bins],
        )

        # Normalise
        # Yielding crossings per second
        cyl_hist_data = np.where(
            residence_cyl != 0, cyl_hist_data / residence_cyl, np.nan
        )

        # Multiply by 3600 to get crossings per hour
        cyl_hist_data *= 3600

        # Normalise histogram to sum to 1
        cyl_hist_data /= np.nansum(cyl_hist_data)

        hist_data.append(cyl_hist_data)

    return hist_data


if __name__ == "__main__":
    main()
