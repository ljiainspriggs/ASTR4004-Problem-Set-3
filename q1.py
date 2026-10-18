
###########################################
### ASTR4004 Problem Set 3 - Question 1 ###
###########################################

## written by LJ Spriggs (u7916719) 

## (a)

# required imports
import sys
import numpy as np

# define function for IMF(M)
def imf(mass):

    # ensure input number/array of numbers
    try:
        m = np.asarray(mass, dtype=float)
    except (ValueError, TypeError):
        sys.exit("Error: Input must be a number or numeric array.")

    # apply Chabrier 2005 IMF
    imf = np.zeros_like(m) # array of zeros to overwrite w IMF function

    lo_m = (m <= 1) # low-mass mask for vectorised processing
    hi_m = (m > 1) # high-mass mask

    if np.any(lo_m): # run for low-mass stars
        exp_input = -1 * ((np.log10(m[lo_m]) - np.log10(0.2))**2) / 0.605
        imf[lo_m] = (0.040 / m[lo_m]) * np.exp(exp_input)

    if np.any(hi_m): # run for high-mass stars
        imf[hi_m] = 0.018 * (m[hi_m])**(-2.35)

    return imf

## (b)

# define function for log-spaced IMF grid
def discrete_imf(m_min=1e-2, m_max=1e2, n_bins=50):

    # edges of uniformly-spaced grid in log-space between m_min and m_max
    # note: n_bins boxes has n_bins + 1 edges
    bin_edges = np.logspace(np.log10(m_min), np.log10(m_max), n_bins+1)

    # IMF at bin_edges masses
    imf_edges = imf(bin_edges)

    return bin_edges, imf_edges

## (c)

# required imports
import matplotlib.pyplot as plt

# define function to generate log-log plot of IMF values
def plot_imf(bin_edges, imf_edges):

    # setup plot
    plt.figure(figsize=(8, 6))

    # plot values
    plt.loglog(bin_edges, imf_edges, marker="^", markersize=2,
               ls="-", lw=0.8, c="DarkRed", label="Chabrier 2005 IMF")

    # labels
    plt.title(f"Discretised IMF (Chabrier 2005) with {len(bin_edges)-1:.0f} Bins", fontsize=14)
    plt.xlabel(r"Mass [$M_\odot$]", fontsize=12)
    plt.ylabel(r"No. Stars per Mass Bin [$M_\odot^{-1}$]", fontsize=12)

    # display details
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # save plot
    plt.savefig("q1c_figure.png", dpi=200)
    plt.show()

## (d)

# define function to extract mode of IMF array
def max_imf(imf_array):
    index = np.argmax(imf_array)
    return imf_array[index]

## (e)

# define function to numerically compute average mass
def avg_mass(bin_edges, imf_edges):

    # extract bin widths from edges
    bin_widths = np.diff(bin_edges)

    # compute arithmetic mean for each bin
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    imf_centres = 0.5 * (imf_edges[1:] + imf_edges[:-1])

    # normalise IMF to PDF
    bin_areas = imf_centres * bin_widths
    integral = np.sum(bin_areas)
    imf_pdf = bin_areas / integral

    # sum to derive average mass
    avg_mass = np.sum(bin_centres * imf_pdf)

    return avg_mass

## (f)

# required imports
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# define function that converges on mass at within 1% accuracy
def find_convergent_mass(target_percent_err=1, low_memory_max=False, keep_array=False):

    # low_memory_max must be False or an integer
    assert (low_memory_max is False) or isinstance(low_memory_max, int), "Error: low_memory_max must be False or integer."

    # set "true" value computed with 5e4 bins for high memory
    if low_memory_max==False:
        true_bin_edges, true_imf_edges = discrete_imf(n_bins=int(5e4))
    # else use max defined by input
    else:
        true_bin_edges, true_imf_edges = discrete_imf(n_bins=low_memory_max)

    true_mass = avg_mass(true_bin_edges, true_imf_edges)
    n_true = len(true_bin_edges) - 1

    if (keep_array is True):
        n_list = []
        mass_list = []

    # initialise min and max bin numbers
    lo_n = 1
    if low_memory_max is False:
        hi_n = int(1e4)
    else:
        hi_n = int(min(1e4, low_memory_max))

    # convert to log boundaries
    lo_log = np.log10(lo_n)
    hi_log = np.log10(hi_n)

    # param to determine log-space resolution limit
    old_n = 0

    if (keep_array is True):
        test_bin_edges, test_imf_edges = discrete_imf(n_bins=lo_n)
        test_mass = avg_mass(test_bin_edges, test_imf_edges)
        n_list.append(lo_n)
        mass_list.append(test_mass)

        test_bin_edges, test_imf_edges = discrete_imf(n_bins=hi_n)
        test_mass = avg_mass(test_bin_edges, test_imf_edges)
        n_list.append(hi_n)
        mass_list.append(test_mass)

    # convert percentage errpr to fractional error
    target_frac_err = 1e-2 * target_percent_err

    # continue when the min and max number of bins are still distinct
    # implement binary searching for efficiency
    while lo_log <= hi_log:

        # test middle of testing region
        mid_log = (lo_log + hi_log) / 2.0 # central n bins in log space
        mid_n = int(np.round(10**mid_log)) # bin number must be integer

        # if rounded n bins unchanged from previous, exit loop w current params
        if mid_n == old_n:
            break
        old_n = mid_n

        # compute avg mass at mid bin size
        test_bin_edges, test_imf_edges = discrete_imf(n_bins=mid_n)
        test_mass = avg_mass(test_bin_edges, test_imf_edges)


        if (keep_array is True):
            n_list.append(mid_n)
            mass_list.append(test_mass)

        # compute fractional error
        frac_err = np.abs(test_mass - true_mass) / true_mass

        # when within 1% err, note values, test lower bin number
        if frac_err <= target_frac_err:
            # note params
            n_optimal = mid_n
            final_mass = test_mass
            # reset bin no limit
            if mid_n > 1:
                hi_log = np.log10(mid_n - 1) # search lower half of log-parameter-space
            else:
                hi_log = lo_log - 1 # break while condition

        # outside of 1% err, true higher bin no
        else: 
            lo_log = np.log10(mid_n + 1)

    if (keep_array is True):
        n_array = np.array(n_list)
        mass_array = np.array(mass_list)
        sorted_indices = np.argsort(n_array)
        return n_optimal, final_mass, n_true, true_mass, n_array[sorted_indices], mass_array[sorted_indices]

    return n_optimal, final_mass, n_true, true_mass


## execute script only when executed directly (not imported)
if __name__ == "__main__":

    # set no. bins param
    n_bins = 50

    # generate log-spaced grid with IMF values
    grid_edges, imf_grid_edges = discrete_imf(m_min=1e-2, m_max=1e2, n_bins=n_bins)
    print("Log-spaced grid successfully generated.")

    # generate IMF plot
    plot_imf(grid_edges, imf_grid_edges)

    # compute mode IMF
    mode = max_imf(imf_grid_edges)
    print(f"Most Frequent Stellar Mass: {mode:.2f} Msun")

    # compute average mass
    avg = avg_mass(grid_edges, imf_grid_edges)
    print(f"Average Stellar Mass: {avg:.2f} Msun")

    # converge on 1% error mass estimate
    n_optimal, final_mass, n_true, true_mass, n_array, mass_array = find_convergent_mass(keep_array=True)
    print(f"Optimal No. Bins: {n_optimal:.0f}")
    print(f"Average Mass at {n_optimal:.0f} Bins: {final_mass:.4f}")
    print(f"True Average Mass at {n_true:.0f} Bins: {true_mass:.4f}")

    # plot avg mass vs n bins
    plt.figure(figsize=(8, 6))
    plt.loglog(n_array, mass_array, marker="^", markersize=2, ls="-", lw=0.8, c="DarkRed", label="Approx. Avg Mass")
    plt.axhline(true_mass, color="black", ls=":", alpha=0.5, label="True Avg Mass")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title("Numerically-Approximated Average Mass vs Bin Number", fontsize=14)
    plt.xlabel("No. Bins", fontsize=12)
    plt.ylabel(r"Average Stellar Mass [$M_\odot$]", fontsize=12)
    plt.legend(loc="lower left", fontsize=8)

    # zoomed-in plot about covnergence region
    ax_inset = plt.axes([0.5, 0.5, 0.35, 0.32])
    ax_inset.loglog(n_array, mass_array, marker="^", markersize=3, ls="-", lw=1.0, c="DarkRed")
    ax_inset.axhline(true_mass, color="black", ls=":", alpha=0.5)
    ax_inset.plot(n_optimal, final_mass, marker="o", color="DarkRed", markersize=6, label="Optimal No. Bins")
    x_min = n_optimal / 1.2
    x_max = n_optimal * 1.2
    mask_zoomed = (n_array >= x_min) & (n_array <= x_max)
    y_min = np.min(mass_array[mask_zoomed]) * 0.98
    y_max = np.max(mass_array[mask_zoomed]) * 1.02
    ax_inset.set_xlim(x_min, x_max)
    ax_inset.set_ylim(y_min, y_max)
    ax_inset.grid(True, which="both", ls="--", alpha=0.2)
    ax_inset.set_title("Convergence Region", fontsize=9)

    # save plot
    plt.savefig("q1f_figure.png", dpi=200, bbox_inches="tight") # alternative for tight_layout() when using axes
    plt.show()