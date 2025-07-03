import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import spiceypy as spice
from hermpy import mag, plotting, utils

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
wong_colours_list = list(wong_colours.values())

orbits = []

number_of_orbits = 3

start_times = [
    dt.datetime(year=2013, month=1, day=1, hour=0) + i * dt.timedelta(days=15)
    for i in range(number_of_orbits)
]
end_times = [
    dt.datetime(year=2013, month=1, day=1, hour=8) + i * dt.timedelta(days=15)
    for i in range(number_of_orbits)
]

for start, end in zip(start_times, end_times):
    orbit_data = mag.Load_Between_Dates(utils.User.DATA_DIRECTORIES["MAG"], start, end)

    orbits.append(orbit_data)


# These positions can then be plotted
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

for i, data in enumerate(orbits):
    axes[0].plot(
        data["X MSM' (radii)"],
        data["Y MSM' (radii)"],
        color=wong_colours_list[i + 1],
        label=f"Orbit on {start_times[i].date()}",
        zorder=10,
    )
    axes[1].plot(
        data["X MSM' (radii)"],
        data["Z MSM' (radii)"],
        color=wong_colours_list[i + 1],
        zorder=10,
    )


axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.2), ncol=3)

planes = ["xy", "xz"]
hemisphere = ["left", "left"]
for i, ax in enumerate(axes):
    plotting.Plot_Mercury(
        axes[i], shaded_hemisphere=hemisphere[i], plane=planes[i], frame="MSM"
    )
    plotting.Add_Labels(axes[i], planes[i], frame="MSM'")
    plotting.Plot_Magnetospheric_Boundaries(ax, plane=planes[i])
    plotting.Square_Axes(ax, 6)

plt.savefig("./figures/fig01_trajectories_example.pdf", format="pdf")
