import argparse
import os
import struct
import mmap

from netCDF4 import Dataset
from tqdm import tqdm

from ensembles_old import Config, Ensemble, EnsembleFormatError, EnsembleWriter

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

    fname = f"{path[:-4]}.nc"

    # Set up structure of HDF5 file
    config_file(fname, n_cells, len(ens_indexes), cfg)

    # List for storing ensembles, add first ensemble
    batch = []
    batch.append(ens)

    writer = EnsembleWriter(fname, batch_size, n_cells)

    with Dataset(fname, "a") as ds:
        writer.initialize_variables(ds)
        # For all ensembles, create object and write batches to file
        skipped = 0
        for i in range(1, len(ens_indexes)):
            try:
                ens_dat = mm[ens_indexes[i]:ens_indexes[i] + numbytes]
                ens = Ensemble.from_bytes(ens_dat)
                batch.append(ens)
            except (EnsembleFormatError, IndexError, ValueError, struct.error) as e:
                skipped += 1
                # tqdm.write(f"Skipped ensemble at index {i}: {e}")
                continue
            finally:
                decode_progress.update(1)

            if len(batch) == batch_size:
                writer.write_batch(batch)
                batch[:] = []
                batch_progress.update(1)

        # Write any leftovers
        if batch:
            writer.write_batch(batch)
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
    with Dataset(fname, "w", format="NETCDF4") as ds:
        # Dimensions
        ds.createDimension("ensemble", n_ens)
        ds.createDimension("cell", n_cells)
        ds.createDimension("beam", 3)
        
        # Config data
        cfg_group = ds.createGroup("config")
        cfg_group.name_str = "vadcp"
        cfg_group.sourceprog = "instrument"
        cfg_group.prog_ver = cfg.prog_ver
        cfg_group.config_str = cfg.config
        cfg_group.n_beams = cfg.n_beams
        cfg_group.n_cells = cfg.n_cells
        cfg_group.pings_per_ensemble = cfg.pings_per_ensemble
        cfg_group.cell_size = cfg.cell_size
        cfg_group.blank = cfg.blank
        cfg_group.corr_threshold = cfg.corr_threshold
        cfg_group.n_codereps = cfg.n_codereps
        cfg_group.evel_threshold = cfg.evel_threshold
        cfg_group.time_between_ping_groups = cfg.time_between_ping_groups
        cfg_group.coord = cfg.coord
        cfg_group.sensors_src = cfg.sensors_src
        cfg_group.sensors_avail = cfg.sensors_avail
        cfg_group.bin1_dist = cfg.bin1_dist
        cfg_group.fls_target_threshold = cfg.fls_target_threshold
        cfg_group.xmit_lag = cfg.xmit_lag
        cfg_group.bandwidth = cfg.bandwidth
        cfg_group.syspower = cfg.syspower
        cfg_group.sernum = cfg.sernum

        # Ranges as a variable along cell dimension
        ranges_var = cfg_group.createVariable(
            "ranges", "f4", ("cell",)
        )
        ranges_var[:] = cfg.ranges.astype("float32")

        # Ensemble data
        ens_group = ds.createGroup("ensembles")

        # Helper for creating variables with compression and metadata
        def create_ens_var(name, dtype, dims, chunks, units=None, long_name=None, scale_factor=None):
            var = ens_group.createVariable(
                name,
                dtype,
                dims,
                zlib=True,
                chunksizes=chunks
            )
            if units is not None:
                var.units = units
            if long_name is not None:
                var.long_name = long_name
            if scale_factor is not None:
                var.vadcp_scale_factor = scale_factor
            return var

        # 1D ensemble variables
        create_ens_var("number", "u4", ("ensemble",), (batch_size,),
                       units="1", long_name="Ensemble Number")

        create_ens_var("mtime", "u4", ("ensemble",), (batch_size,),
                       units="seconds", long_name="Time")

        create_ens_var("depth", "u2", ("ensemble",), (batch_size,),
                       units="meters", long_name="Transducer Depth", scale_factor=0.1)

        create_ens_var("salinity", "i2", ("ensemble",), (batch_size,),
                       units="ppt", long_name="Water Salinity")

        create_ens_var("temperature", "i2", ("ensemble",), (batch_size,),
                       units="degrees C", long_name="Water Temperature", scale_factor=0.01)

        create_ens_var("mpt", "u4", ("ensemble",), (batch_size,),
                       units="seconds", long_name="Sleep Duration", scale_factor=0.01)

        create_ens_var("voltage", "u1", ("ensemble",), (batch_size,),
                       units="volts", long_name="Battery Voltage", scale_factor=0.157)

        # 2D variables (ensemble, cell)
        create_ens_var("x_vel", "i2", ("ensemble", "cell"), (batch_size, n_cells),
                       units="m/s", long_name="X Horizontal Velocity", scale_factor=0.001)

        create_ens_var("y_vel", "i2", ("ensemble", "cell"), (batch_size, n_cells),
                       units="m/s", long_name="Y Horizontal Velocity", scale_factor=0.001)

        create_ens_var("z_vel", "i2", ("ensemble", "cell"), (batch_size, n_cells),
                       units="m/s", long_name="Z Vertical Velocity", scale_factor=0.001)

        # 3D variables (ensemble, cell, beam)
        create_ens_var("corr", "u1", ("ensemble", "cell", "beam"), (batch_size, n_cells, 3),
                       units="1", long_name="Correlation Magnitude")

        create_ens_var("intens", "u1", ("ensemble", "cell", "beam"), (batch_size, n_cells, 3),
                       units="1", long_name="Echo Intensity")

        create_ens_var("perc_good", "u1", ("ensemble", "cell", "beam"), (batch_size, n_cells, 3),
                       units="percent", long_name="Percentage of Good Pings")

        create_ens_var("surface_track", "u4", ("ensemble",), (batch_size,),
                       units="meters", long_name="Corrected Depth from Surface Track", scale_factor=0.0001)

        create_ens_var("surface_track_uncorr", "u4", ("ensemble",), (batch_size,),
                       units="meters", long_name="Uncorrected Depth from Surface Track", scale_factor=0.0001)

        create_ens_var("v_amp", "u1", ("ensemble",), (batch_size,),
                       units="1", long_name="Signal Amplitude at Surface")

        create_ens_var("v_pgood", "u1", ("ensemble",), (batch_size,),
                       units="percent", long_name="Percentage Good of Surface Track")


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
