import argparse
import os
import struct
import mmap

import h5py
from tqdm import tqdm

from ensembles import Config, Ensemble, EnsembleFormatError, EnsembleWriter

# Variable for batch writing size. Value was determined by benchmarking.
# With 4096 max sized ensembles that would be ~3MB of data in RAM, which any modern system should be capable of
batch_size = 4096


# Convert file to readable data
def decode_bin(path):
    file = open(path, "rb")
    # Get file size for progress bar
    file_size = os.path.getsize(path)
    
    # Create a memory map for faster access than just the file
    mm = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)
    file.close()

    # List for storing start index of each ensamble
    ens_indexes = []
    current_offset = 0

    # Set up progress bar
    scan_progress = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Scanning file"
    )

    # Search the entire file for ensemble headers
    while current_offset < file_size:
        # not enough bytes left for a header
        if current_offset + 4 > file_size:
                break
        
        # Get header ID, source ID, and numbytes
        header = mm[current_offset:current_offset + 4]

        # If the two start bytes are right (two 7f bytes in a row)
        if header[0] == 0x7f and header[1] == 0x7f:
            # Get ensemble size
            ens_size = header[2] + (header[3] << 8) + 2

            # If size is within expected bounds (removes potential errors from random 7f7f data)
            if 32 <= ens_size <= 4096 and current_offset + ens_size <= file_size:
                ens_indexes.append(current_offset)
                current_offset += ens_size
                scan_progress.update(ens_size)
                continue

        # Continue seeking
        current_offset += 1
        scan_progress.update(1)

    # Needed to finish scan bar since never actually reaches last byte of file
    scan_progress.n = scan_progress.total + 1
    scan_progress.close()

    # Progress bar for decoding and writing batches
    decode_progress = tqdm(
        total=len(ens_indexes),
        unit=" Ensembles",
        desc="Decoding Ensembles",
    )

    batch_progress = tqdm(
        total=len(ens_indexes) // batch_size + 1,
        unit=" Batches",
        desc="Writing Batches"
    )

    # Get the length of an ensemble
    offset = ens_indexes[0]
    numbytes = struct.unpack("<h", mm[offset + 2:offset + 4])[0]

    # Decode the first ensemble to get config and structure data
    ens_dat = mm[offset:offset + numbytes]
    ens = Ensemble.from_bytes(ens_dat)
    decode_progress.update(1)

    if not Ensemble.config:
        raise EnsembleFormatError(
            "Configuration data missing from first ensemble")
    cfg = Ensemble.config

    # Needed for velocity shapes
    n_cells = Ensemble.config.n_cells

    fname = f"{path[:-4]}.hdf5"

    # Set up structure of HDF5 file
    config_file(fname, n_cells, len(ens_indexes), cfg)

    # List for storing ensembles, add first ensemble
    batch = []
    batch.append(ens)

    writer = EnsembleWriter(fname, batch_size, n_cells)

    with h5py.File(fname, "a") as f:
        writer.initialize_datasets(f)
        # For all ensembles, create object and write batches to file
        skipped = 0
        for i in range(1, len(ens_indexes)):
            try:
                ens_dat = mm[ens_indexes[i]:ens_indexes[i] + numbytes]
                ens = Ensemble.from_bytes(ens_dat)
                batch.append(ens)
            except (EnsembleFormatError, IndexError, ValueError, struct.error) as e:
                skipped += 1
                # You can uncomment the next line for debugging:
                # tqdm.write(f"Skipped ensemble at index {i}: {e}")
                continue
            finally:
                decode_progress.update(1)

            if len(batch) == batch_size:
                writer.write_batch(batch, f)
                batch[:] = []
                batch_progress.update(1)

        # Write any leftovers
        if batch:
            writer.write_batch(batch, f)
            batch[:] = []
            batch_progress.update(1)

        decode_progress.set_description(f"Decoding Ensembles (Skipped {skipped})")

    # Extra updates to force bars to render properly
    decode_progress.update(1)
    batch_progress.update(1)
    decode_progress.close()
    batch_progress.close()
    
    mm.close()


def config_file(fname: str, n_cells: int, n_ens: int, cfg: Config):
    """Helper function for creating file to clean up main function a bit"""
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
        cfg_group.create_dataset("ranges", data=cfg.ranges, dtype="float")

        # Ensemble data
        ens_group = f.create_group("ensembles")

        ds = ens_group.create_dataset("number", shape=(
            n_ens,), dtype="uint32", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="1", long_name="Ensemble Number")

        ds = ens_group.create_dataset("mtime", shape=(
            n_ens,), dtype="uint32", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="seconds", long_name="Time")

        ds = ens_group.create_dataset("depth", shape=(
            n_ens,), dtype="uint16", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="meters",
                     long_name="Transducer Depth", scale_factor=0.1)

        ds = ens_group.create_dataset("salinity", shape=(
            n_ens,), dtype="int16", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="ppt", long_name="Water Salinity")

        ds = ens_group.create_dataset("temperature", shape=(
            n_ens,), dtype="int16", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="degrees C",
                     long_name="Water Temperature", scale_factor=0.01)

        ds = ens_group.create_dataset("mpt", shape=(
            n_ens,), dtype="uint32", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="seconds",
                     long_name="Sleep Duration", scale_factor=0.01)

        ds = ens_group.create_dataset("voltage", shape=(
            n_ens,), dtype="uint8", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="volts",
                     long_name="Battery Voltage", scale_factor=0.157)

        ds = ens_group.create_dataset("x_vel", shape=(n_ens, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells), compression="gzip")
        add_metadata(ds, units="m/s",
                     long_name="X Horizontal Velocity", scale_factor=0.001)

        ds = ens_group.create_dataset("y_vel", shape=(n_ens, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells), compression="gzip")
        add_metadata(ds, units="m/s",
                     long_name="Y Horizontal Velocity", scale_factor=0.001)

        ds = ens_group.create_dataset("z_vel", shape=(n_ens, n_cells), dtype="int16", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells), compression="gzip")
        add_metadata(ds, units="m/s",
                     long_name="Z Vertical Velocity", scale_factor=0.001)

        ds = ens_group.create_dataset("corr", shape=(n_ens, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3), compression="gzip")
        add_metadata(ds, units="1", long_name="Correlation Magnitude")

        ds = ens_group.create_dataset("intens", shape=(n_ens, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3), compression="gzip")
        add_metadata(ds, units="1", long_name="Echo Intensity")

        ds = ens_group.create_dataset("perc_good", shape=(n_ens, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3), compression="gzip")
        add_metadata(ds, units="percent", long_name="Percentage of Good Pings")

        ds = ens_group.create_dataset("surface_track", shape=(
            n_ens,), dtype="uint32", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="meters",
                     long_name="Corrected Depth from Surface Track", scale_factor=0.0001)

        ds = ens_group.create_dataset("surface_track_uncorr", shape=(
            n_ens,), dtype="uint32", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="meters",
                     long_name="Uncorrected Depth from Surface Track", scale_factor=0.0001)

        ds = ens_group.create_dataset("v_amp", shape=(
            n_ens,), dtype="uint8", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="1", long_name="Signal Amplitude at Surface")

        ds = ens_group.create_dataset("v_pgood", shape=(
            n_ens,), dtype="uint8", chunks=batch_size, maxshape=(None,), compression="gzip")
        add_metadata(ds, units="percent",
                     long_name="Percentage Good of Surface Track")


def add_metadata(
    dataset: h5py.Dataset,
    units: str | None = None,
    long_name: str | None = None,
    scale_factor: float | None = None
):
    """Helper function for adding metadata to datasets"""
    if units:
        dataset.attrs["units"] = units
    if long_name:
        dataset.attrs["long_name"] = long_name
    if scale_factor:
        dataset.attrs["scale_factor"] = scale_factor


def main():
    # Command line argument for file path
    parser = argparse.ArgumentParser(
        prog="DecodeADCP",
        description="Parses VADCP output data and decodes it"
    )

    parser.add_argument("path", help="Path to data file")
    args = parser.parse_args()

    # Get path from arguments, make sure file exists
    path = args.path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Invalid Path Provided {path}")

    decode_bin(path)


if __name__ == "__main__":
    main()
