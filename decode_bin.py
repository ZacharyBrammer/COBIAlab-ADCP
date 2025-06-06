import argparse
import gc
import os
import struct

import h5py

from ensembles import Ensemble, EnsembleFormatError, EnsembleWriter

# Command line argument for file path
parser = argparse.ArgumentParser(
    prog="DecodeADCP",
    description="Parses VADCP output data and decodes it"
)

parser.add_argument(dest="path", help="Path to data file")
args = parser.parse_args()

# Get path from arguments, make sure file exists
path = args.path
if os.path.exists(path):
    pass
else:
    raise FileNotFoundError("Invalid Path Provided")

# Variable for batch writing size
# TODO: Make change based on available memory, will do calculations based on max ensemble size of 2070B
batch_size = 1024


# Convert file to readable data
def decode_bin(path):
    file = open(path, "rb")

    # List for storing start index of each ensamble
    ens_indexes = []
    current_offset = 0

    # Search the entire file for ensemble headers
    while True:
        # Get header ID, source ID, and numbytes
        header = file.read(4)

        # End of file check
        if len(header) < 4:
            break

        # If the two start bytes are right (two 7f bytes in a row)
        if header[0] == 0x7f and header[1] == 0x7f:
            # Get ensemble size
            ens_size = header[2] + (header[3] << 8) + 2

            # If size is within expected bounds (removes potential errors from random 7f7f data)
            if 32 <= ens_size <= 4096:
                ens_indexes.append(current_offset)
                current_offset += ens_size
                file.seek(current_offset)
                continue
            else:
                # Continue seeking
                current_offset += 1
                file.seek(current_offset)
        else:
            # Continue seeking
            current_offset += 1
            file.seek(current_offset)

    print(f"{len(ens_indexes)} ensembles found, processing")

    # Reset to head of file
    file.seek(0, 0)

    # Get the length of an ensemble
    file.seek(ens_indexes[0] + 2, 0)
    numbytes = struct.unpack("<h", file.read(2))[0]

    # Decode the first ensemble to get config and structure data
    file.seek(ens_indexes[0], 0)
    ens_dat = file.read(numbytes)
    ens = Ensemble.from_bytes(ens_dat)

    if not Ensemble.config:
        raise EnsembleFormatError(
            "Configuration data missing from first ensemble")
    cfg = Ensemble.config

    # Needed for velocity shapes
    n_cells = Ensemble.config.n_cells

    fname = f"{path[:-4]}.hdf5"

    # Set up structure of HDF5 file
    with h5py.File(fname, "w") as f:
        # Config data
        cfg_group = f.create_group("config")
        cfg_group.attrs.create("name", data="vadcp")
        cfg_group.attrs.create("sourceprog", data="instrument")
        cfg_group.attrs.create("prog_ver", data=cfg.prog_ver)
        cfg_group.attrs.create("config", data=cfg.config)
        cfg_group.attrs.create("n_beams", data=cfg.n_beams)
        cfg_group.attrs.create("n_cells", data=cfg.n_cells)
        cfg_group.attrs.create("pings_per_ensemble",
                               data=cfg.pings_per_ensemble)
        cfg_group.attrs.create("cell_size", data=cfg.cell_size)
        cfg_group.attrs.create("blank", data=cfg.blank)
        cfg_group.attrs.create("corr_threshold", data=cfg.corr_threshold)
        cfg_group.attrs.create("n_codereps", data=cfg.n_codereps)
        cfg_group.attrs.create("evel_threshold", data=cfg.evel_threshold)
        cfg_group.attrs.create("time_between_ping_groups",
                               data=cfg.time_between_ping_groups)
        cfg_group.attrs.create("coord", data=cfg.coord)
        cfg_group.attrs.create("sensors_src", data=cfg.sensors_src)
        cfg_group.attrs.create("sensors_avail", data=cfg.sensors_avail)
        cfg_group.attrs.create("bin1_dist", data=cfg.bin1_dist)
        cfg_group.attrs.create("fls_target_threshold",
                               data=cfg.fls_target_threshold)
        cfg_group.attrs.create("xmit_lag", data=cfg.xmit_lag)
        cfg_group.attrs.create("bandwidth", data=cfg.bandwidth)
        cfg_group.attrs.create("syspower", data=cfg.syspower)
        cfg_group.attrs.create("sernum", data=cfg.sernum)
        ds = cfg_group.create_dataset("ranges", data=cfg.ranges)

        # Ensemble data
        # TODO: dataset.attrs["units"], dataset.attrs["scale_factor"] where required
        # TODO: also, fix datatypes
        ens_group = f.create_group("ensembles")
        ens_group.create_dataset("number", shape=(
            0,), dtype="uint16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("mtime", shape=(
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("depth", shape=(
            0,), dtype="uint16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("salinity", shape=(
            0,), dtype="int16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("temperature", shape=(
            0,), dtype="int16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("mpt", shape=(
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("voltage", shape=(
            # Scaling needs to be 0.157
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("x_vel", shape=(0, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("y_vel", shape=(0, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("z_vel", shape=(0, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("corr", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("intens", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("perc_good", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("surface_track", shape=(
            # Scaling needs to be 0.0001
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("surface_track_uncorr", shape=(
            # Same scaling
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("v_amp", shape=(
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("v_pgood", shape=(
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,))

    # List for storing ensembles, add first ensemble
    batch = []
    batch.append(ens)

    writer = EnsembleWriter(fname, batch_size, n_cells)

    # For all ensembles, create object and write batches to file
    for i in range(1, len(ens_indexes)):
        file.seek(ens_indexes[i], 0)
        ens_dat = file.read(numbytes)
        ens = Ensemble.from_bytes(ens_dat)
        batch.append(ens)

        if len(batch) == batch_size:
            writer.write_batch(batch)
            # Clear batch list and free up memory
            batch.clear()
            gc.collect()

    # Write any leftover batches
    if batch:
        writer.write_batch(batch)
        # Clear batch list and free up memory
        batch.clear()
        gc.collect()


decode_bin(path)
