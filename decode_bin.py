import argparse
import gc
import os
import struct

import h5py
import numpy as np

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

    # Get first data batch
    buffer_size = int(1e6)
    eoe = 127

    # Get the end of the file (needed for when file is larger than buffer size)
    file.seek(0, 2)
    eof = file.tell()
    file.seek(0, 0)  # Reset position

    # Read a batch of data
    batch = np.fromfile(file, dtype="uint8", count=buffer_size)

    # Appending with numpy arrays is inefficient so use a list
    increment = list(np.fromfile(file, dtype="uint8", count=2))

    # Search until the next ensemble starts so nothing gets split
    while (len(increment) < 2 or (increment[-1] != eoe and increment[-2] != eoe)) and (file.tell() != eof):
        # Read the next byte
        next_byte = file.read(1)

        # If byte is empty, break from loop
        if not next_byte:
            break
        increment.append(next_byte[0])

    # Convert increment to ndarray and add to data
    increment = np.array(increment, dtype="uint8")
    data = np.concatenate((batch, increment[:-1]))

    # List for storing start index of each ensamble
    ens_indexes = []

    # Offset for later batches
    cumulative_index = 0

    # Search entire file for ensemble headers
    while data.size > 0:
        # Get index of all 7f values in the data
        potentialIndexes = np.where(data == 0x7f)[0]

        # Check find where there are two consecutive 7f values indicating an ensemble header
        headers = (np.diff(potentialIndexes) == 1)
        headers = np.append(headers, False)

        # Get index of all headers
        headerIndexes = potentialIndexes[headers]
        headerIndexes += cumulative_index

        # Add header indexes to list, update offset
        ens_indexes.extend(headerIndexes.tolist())
        cumulative_index += data.size

        print(f"{len(data)} bytes scanned with {len(headerIndexes)} ensembles.")

        # Check the rest of the file by updating data
        # Read a batch of data
        batch = np.fromfile(file, dtype="uint8", count=buffer_size)

        # Appending with numpy arrays is inefficient so use a list
        increment = list(np.fromfile(file, dtype="uint8", count=2))

        # Search until the next ensemble starts so nothing gets split
        while (len(increment) < 2 or (increment[-1] != eoe and increment[-2] != eoe)) and (file.tell() != eof):
            # Read the next byte
            next_byte = file.read(1)

            # If byte is empty, break from loop
            if not next_byte:
                break
            increment.append(next_byte[0])

        # Convert increment to ndarray and add to data
        increment = np.array(increment, dtype="uint8")
        data = np.concatenate((batch, increment[:-1]))

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
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,)) # Scaling needs to be 0.157
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
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,)) # Scaling needs to be 0.0001
        ens_group.create_dataset("surface_track_uncorr", shape=(
            0,), dtype="uint32", chunks=batch_size, maxshape=(None,)) # Same scaling
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
