
###########################################
### ASTR4004 Problem Set 3 - Question 3 ###
###########################################

## written by LJ Spriggs (u7916719) 

## (a)

# required imports
import numpy as np
from astropy.io import fits

# read-in .fits data
# ensure file in same dir as this script
def readin_fits(filename="brick"):
    with fits.open(f"./{filename}.fits") as file:
        file.info() # print file details/structure
        data = file[0].data

    return data

## (b)

# required imports
import matplotlib.pyplot as plt

# define function for plotting simple cmap
def col_dens_map(data, filename="brick"):

    # plot setup
    plt.figure(figsize=(6, 8))
    plt.title("column density map")

    # plot data
    plt.imshow(data, aspect="auto", cmap="jet")
    plt.colorbar()

    # save plot
    plt.tight_layout()
    plt.savefig(f"./{filename}.pdf")
    plt.show()

## (c)

# define function for plotting doubly-mirrored image
def mirror_col_dens(data, filename):

    # mirrored data setup
    no_mir = np.array(data)
    hori_mir = np.fliplr(no_mir)
    vert_mir = np.flipud(no_mir)
    full_mir = np.flipud(hori_mir)

    # stitch array as in thumbnail
    stitched_arr = np.block([
        [no_mir, hori_mir],
        [vert_mir, full_mir] 
    ])

    # plot
    plt.figure(figsize=(6, 8))
    plt.title("mirrored column density map")
    plt.imshow(stitched_arr, aspect="auto", cmap="jet")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"./{filename}_mirrored.pdf")
    plt.show()

    return stitched_arr

## (d)

def pad_square(data, target_dim=1278, filename="brick"):

    # derive padding values
    v, h = data.shape
    pad_v = target_dim - v
    pad_h = target_dim - h
    
    # handle odd values for top and bottom padding
    pad_bottom = pad_v // 2 # integer division
    pad_top = pad_v - pad_bottom
    
    # handle odd values for left and right padding
    pad_left = pad_h // 2 
    pad_right = pad_h - pad_left
    
    # Apply symmetric zero-padding using np.lib.pad
    padded_arr = np.pad(data, pad_width=((pad_bottom, pad_top), (pad_left, pad_right)), 
                         mode='constant', constant_values=0)

    # plot
    plt.figure(figsize=(8, 8))
    plt.title("mirrored and zero-padded column density map")
    plt.imshow(padded_arr, aspect="auto", cmap="jet")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"./{filename}_mirrored_zped.pdf")
    plt.show()
    
    return padded_arr

## (e)

# required packages
from pyfftw.builders import fft2

# define function to compute threaded FFTW
def compute_fftw(data, n_threads=2, filename="brick"):

    # compute fourier transform
    fft_builder = fft2(data, threads=n_threads)
    raw_ft = fft_builder() # execute the fourier transform

    # shift k = (0, 0) from beginning to centre of array
    shift_ft = np.fft.fftshift(raw_ft)

    # take log10 of power spectrum
    power = np.abs(shift_ft)**2
    power[power == 0] = 1e-10 # avoid log(0)
    log_power = np.log10(power)

    # plot
    plt.figure(figsize=(8, 8))
    plt.title("fourier image column density map")
    plt.imshow(log_power, aspect="auto", cmap="jet", vmin=13, vmax=57)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"./{filename}_fourier_image.pdf")
    plt.show()

    return shift_ft

## (f)

# define function to calculate 1D radially-averaged power spectrum
def radial_power_spec(ft_data, max_dim=1278, filename="brick"):

    # setup linear k array as per assignment instructions
    kxy = np.linspace(-max_dim/2, max_dim/2-1, num=max_dim)
    k1 = np.zeros((max_dim, max_dim))
    k2 = np.zeros((max_dim, max_dim))
    for n in range(0, max_dim):
        k1[n, :] = kxy
        k2[:, n] = kxy
    k_image = np.sqrt(k1*k1 + k2*k2)
    
    # define empty power spectrum to overwrite
    n_bins = int(max_dim // 2 + 1) # integer division
    Pk = np.zeros(n_bins)
    k = np.linspace(0, max_dim/2, num=n_bins)
    
    # power array as in part (e)
    power_2d = np.abs(ft_data)**2
    
    # loop over all shells
    for i, ik in enumerate(k): # take k bin index and radii distance of k
        k_shell_ind = np.where((k_image >= ik) & (k_image < ik+1)) # find indices in k shell
        Pk[i] = np.sum(power_2d[k_shell_ind]) # sum all power in k shell
    
    return k, Pk

## (g)

# define function to plot radially-average power spectrum
def plot_powerspec(wavenumber_arr, power_arr, filename="brick"):

    # ensure numpy arrays
    wavenumber_arr = np.array(wavenumber_arr)
    power_arr = np.array(power_arr)

    # plot
    plt.figure(figsize=(8, 6))
    plt.title("1D Radially-Averaged Power Spectrum")
    plt.loglog(wavenumber_arr[1:], power_arr[1:], markersize=0, ls="-", lw=0.8, c="DarkRed") # exclude k=0 to prevent log(0)
    plt.xlabel(r"Wavenumber $k$")
    plt.ylabel(r"Averaged Power $P(k)$")
    plt.tight_layout()
    plt.savefig(f"./{filename}_power_spectrum.pdf")
    plt.show()

## execute script only when executed directly (not imported)
if __name__ == "__main__":

    filename = "brick"

    # read-in data
    data = readin_fits(filename="brick")
    print(f"Data successfully read-in as type {type(data)} with shape {data.shape}.")

    # preliminary cmap plot
    col_dens_map(data, filename="brick")

    # mirrored cmap plot
    mir_arr = mirror_col_dens(data, filename="brick")

    # padded cmap plot
    pad_arr = pad_square(mir_arr, target_dim=1278, filename="brick")

    # FFT cmap plot
    fft_arr = compute_fftw(pad_arr, n_threads=2, filename="brick")

    # radially-averaged power spectrum
    wavenum, power = radial_power_spec(fft_arr, max_dim=1278, filename="brick")

    # plot radially-averaged power spectrum
    plot_powerspec(wavenum, power, filename="brick")