import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from hermpy.plotting import wong_colours

regions = pd.read_csv("./resources/new_regions.csv")

regions = regions.dropna()

# We're not concerned with extremely high duration regions, so we remove
# anything above 3 sigma.
regions = regions[(np.abs(scipy.stats.zscore(regions["Duration (seconds)"])) <= 3)]

# Find knee point
kneedle = kneed.KneeLocator(
    regions["Duration (seconds)"],
    regions["Confidence"],
    curve="concave",
    direction="increasing",
)


def Log_Fit(x, a, b, c):
    return 1 - np.exp(-a * (x - b)) + c


test_pars = [1, 1, 1]
pars, cov = scipy.optimize.curve_fit(
    Log_Fit, regions["Duration (seconds)"], regions["Confidence"], test_pars
)
fit_errors = np.sqrt(np.diag(cov))

fig, ax = plt.subplots(figsize=(8, 6))

_, _, _, hist = ax.hist2d(
    regions["Duration (seconds)"], regions["Confidence"], norm="log", bins=50
)

plt.colorbar(hist, ax=ax, label="Number of regions")

x_range = np.linspace(1, regions["Duration (seconds)"].max(), 1000)
ax.plot(
    x_range,
    Log_Fit(x_range, *pars),
    lw=2,
    color=wong_colours["black"],
    label=r"Least Squares Fit: f(x) = $1 - e^{-a(x - b)} + c$"
    + f"\n    $a = {pars[0]:.4f}$\n    $b = {pars[1]:.2f}$\n    $c = {pars[2]:.4f}$",
)

ax.axvline(
    kneedle.knee,
    color=wong_colours["black"],
    ls="--",
    lw=2,
    label=f"Curve Knee = {kneedle.knee:.0f} seconds",
)
ax.axhline(
    Log_Fit(kneedle.knee, *pars),
    color=wong_colours["black"],
    ls=":",
    lw=2,
    label=f"Fit @ Curve Knee = {Log_Fit(kneedle.knee, *pars):.2f}",
)

ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence [arb.]")

ax.margins(0)
ax.set_xlim(0, x_range[-1])

ax.legend()

plt.savefig(
    "./figures/fig06_region_confidence_vs_duration.pdf",
    format="pdf",
)
