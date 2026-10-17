
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
