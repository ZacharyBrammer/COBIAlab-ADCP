import h5py
from netCDF4 import Dataset
import numpy as np

# >>> EDIT THESE <<<
H5_PATH = "data/Parris_Island_2_VADCP_21May2025.hdf5"    # original HDF5 file with ints
NC_PATH = "data/Parris_Island_2_VADCP_21May2025_d.nc"    # new NetCDF file with floats

# Variables and their scale factors (raw_int * scale -> physical units)
SCALES = {
    "depth": 0.1,                 # uint16 * 0.1 -> meters
    "temperature": 0.01,          # int16  * 0.01 -> deg C
    "mpt": 0.01,                  # uint32 * 0.01 -> seconds
    "voltage": 0.157,             # uint8  * 0.157 -> volts
    "x_vel": 0.001,               # int16  * 0.001 -> m/s
    "y_vel": 0.001,
    "z_vel": 0.001,
    "surface_track": 0.0001,      # uint32 * 1e-4 -> meters
    "surface_track_uncorr": 0.0001,
}

# Numerical tolerance for comparing floats
ATOL = 1e-6
RTOL = 1e-6


def main():
    with h5py.File(H5_PATH, "r") as h5, Dataset(NC_PATH) as ds:
        h5_ens = h5["ensembles"]
        nc_ens = ds.groups["ensembles"]

        print(f"Comparing HDF5: {H5_PATH}")
        print(f"     with NetCDF: {NC_PATH}")
        print("-" * 60)

        for name, scale in SCALES.items():
            print(f"\nVariable: {name}")
            h5_var = h5_ens[name][:]
            nc_var = nc_ens.variables[name][:]

            print(f"  HDF5 shape: {h5_var.shape}, dtype: {h5_var.dtype}")
            print(f"  NetCDF shape: {nc_var.shape}, dtype: {nc_var.dtype}")

            if h5_var.shape != nc_var.shape:
                print("  ❌ Shape mismatch, skipping numerical comparison.")
                continue

            # Scale raw ints to physical units as float64 for comparison
            h5_phys = h5_var.astype("float64") * scale
            nc_phys = nc_var.astype("float64")

            # Basic stats
            print(f"  HDF5 phys min/max: {h5_phys.min():.6g}, {h5_phys.max():.6g}")
            print(f"  NetCDF phys min/max: {nc_phys.min():.6g}, {nc_phys.max():.6g}")

            # Differences
            diff = h5_phys - nc_phys
            max_abs_diff = np.max(np.abs(diff))
            max_rel_diff = np.max(
                np.abs(diff) / (np.abs(h5_phys) + 1e-12)
            )

            allclose = np.allclose(h5_phys, nc_phys, rtol=RTOL, atol=ATOL)

            print(f"  Max abs diff: {max_abs_diff:.3e}")
            print(f"  Max rel diff: {max_rel_diff:.3e}")
            print(f"  np.allclose (rtol={RTOL}, atol={ATOL}): {allclose}")

            # Show first few elements for a quick eyeball
            # Flatten to avoid issues with 2D/3D shapes
            flat_h5 = h5_phys.ravel()
            flat_nc = nc_phys.ravel()

            print("  First 5 HDF5 phys values:", flat_h5[:5])
            print("  First 5 NetCDF phys values:", flat_nc[:5])

            if not allclose:
                # Count elements that differ more than tolerance
                mask = np.abs(diff) > (ATOL + RTOL * np.abs(h5_phys))
                num_bad = np.count_nonzero(mask)
                print(f"  ⚠ {num_bad} elements differ beyond tolerance.")

        print("\nDone.")


if __name__ == "__main__":
    main()
