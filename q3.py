
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

## execute script only when executed directly (not imported)
if __name__ == "__main__":

    filename = "brick"

    # read-in data
    data = readin_fits(filename="brick")
    print(f"Data successfully read-in as type {type(data)} witth shape {data.shape}.")

    # preliminary cmap plot
    col_dens_map(data, filename="brick")