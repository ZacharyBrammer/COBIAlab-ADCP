import argparse
import os
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import ClassVar, Optional

import h5py
import numpy as np

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

    # Needed for velocity shapes
    n_cells = Ensemble.config.n_cells

    # Set up structure of HDF5 file
    with h5py.File(f"{path[:-4]}.hdf5", "w") as f:
        config_group = f.create_group("config")
        # TODO: Set up manually instead of this bs. strings will be dtype=h5py.string_dtype(encoding="ascii", length=20)
        for field, value in asdict(Ensemble.config).items():
            if isinstance(value, np.ndarray):
                config_group.create_dataset(field, data=value)
            else:
                config_group.attrs[field] = value

        ens_group = f.create_group("ensembles")
        ens_group.create_dataset("number", shape=(
            0,), dtype="uint16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("mtime", shape=(
            0,), dtype="float64", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("depth", shape=(
            0,), dtype="uint16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("salinity", shape=(
            0,), dtype="int16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("temperature", shape=(
            0,), dtype="int16", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("mpt", shape=(
            0,), dtype="float64", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("voltage", shape=(
            0,), dtype="float32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("x_vel", shape=(0, n_cells), dtype="float64", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("y_vel", shape=(0, n_cells), dtype="float64", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("z_vel", shape=(0, n_cells), dtype="float64", chunks=(
            batch_size, n_cells), maxshape=(None, n_cells))
        ens_group.create_dataset("corr", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("intens", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("perc_good", shape=(0, n_cells, 3), dtype="uint8", chunks=(
            batch_size, n_cells, 3), maxshape=(None, n_cells, 3))
        ens_group.create_dataset("surface_track", shape=(
            0,), dtype="float32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("surface_track_uncorr", shape=(
            0,), dtype="float32", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("v_amp", shape=(
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,))
        ens_group.create_dataset("v_pgood", shape=(
            0,), dtype="uint8", chunks=batch_size, maxshape=(None,))

    """
    enss = []

    for i in range(len(ens_indexes)):
        file.seek(ens_indexes[i], 0)
        ens_dat = file.read(numbytes)
        ens = Ensemble.from_bytes(ens_dat)
        enss.append(ens)

    for ens in enss:
        print(ens.number)
    """


@dataclass
class Config:
    """Class for storing and decoding config data"""
    name: str = "vadcp"
    sourceprog: str = "instrument"
    prog_ver: float = 0
    config: str = ""
    n_beams: int = 0
    n_cells: int = 0
    pings_per_ensemble: int = 0
    cell_size: float = 0.0
    blank: float = 0.0
    corr_threshold: int = 0
    n_codereps: int = 0
    evel_threshold: int = 0
    time_between_ping_groups: float = 0.0
    coord: str = ""
    sensors_src: str = ""
    sensors_avail: str = ""
    bin1_dist: float = 0.0
    fls_target_threshold: int = 0
    xmit_lag: float = 0.0
    bandwidth: float = 0.0
    syspower: int = 0
    sernum: int = 0
    b_angle: int = 0
    ranges: np.ndarray = field(default_factory=lambda: np.zeros(1))

    @classmethod
    def from_bytes(cls, bytes):
        c = cls()

        prog_ver = struct.unpack("2B", bytes[2:4])
        c.prog_ver = prog_ver[0] + prog_ver[1] / 100
        c.config = f"{format(bytes[5], "08b")}-{format(bytes[4], "08b")}"
        c.n_beams = bytes[8]
        c.n_cells = bytes[9]
        c.pings_per_ensemble = struct.unpack("H", bytes[10:12])[0]
        c.cell_size = struct.unpack("H", bytes[12:14])[0] * 0.01
        c.blank = struct.unpack("H", bytes[14:16])[0] * 0.01
        c.corr_threshold = bytes[17]
        c.n_codereps = bytes[18]
        c.evel_threshold = struct.unpack("H", bytes[20:22])[0]
        time_between_ping_groups = struct.unpack("3B", bytes[22:25])
        c.time_between_ping_groups = np.sum(np.multiply(
            time_between_ping_groups, (60, 1, 0.01)))  # Converts to seconds
        c.coord = format(bytes[25], "08b")
        c.sensors_src = format(bytes[30], "08b")
        c.sensors_avail = format(bytes[31], "08b")
        c.bin1_dist = struct.unpack("H", bytes[32:34])[0] * 0.01
        c.fls_target_threshold = bytes[38]
        c.xmit_lag = struct.unpack("H", bytes[40:42])[0] * 0.01

        # Documentation says this is just byte 51?
        c.bandwidth = struct.unpack("H", bytes[50:52])[0] * 0.01

        c.syspower = bytes[52]
        c.sernum = struct.unpack("I", bytes[54:58])[0]

        # c.b_angle = bytes[58] - I checked the documentation and this is outside the size of the config data and value returned by matlab is just the ID for the next chunk of data

        c.ranges = np.arange(c.n_cells) * c.cell_size + c.bin1_dist

        return c


@dataclass
class Ensemble:
    """Class for storing and decoding ensemble data"""
    config: ClassVar[Optional[Config]] = None
    config_length: int = 58  # Fixed length from operation manual
    datatypes: int = 0
    dat_offsets: tuple = field(default_factory=tuple)
    number: np.uint16 = np.uint16(0)
    mtime: np.float64 = np.float64(0.0)
    depth: np.float32 = np.float32(0.0)
    salinity: np.int16 = np.int16(0)
    temperature: np.float32 = np.float32(0.0)
    mpt: np.float64 = np.float64(0.0)
    voltage: np.float32 = np.float32(0.0)
    x_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64))
    y_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64))
    z_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64))
    corr: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    intens: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    perc_good: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    surface_track: np.float32 = np.float32(0.0)
    surface_track_uncorr: np.float32 = np.float32(0.0)
    v_amp: np.uint8 = np.uint8(0)
    v_pgood: np.uint8 = np.uint8(0)

    @classmethod
    def from_bytes(cls, bytes):
        e = cls()

        # Read header
        # cfgid = struct.unpack("2B", bytes[:2])
        # numbytes = bytes[2:4]
        datatypes = struct.unpack("b", bytes[5:6])[0]
        e.datatypes = datatypes

        dat_offsets = struct.unpack(
            f"<{datatypes}h", bytes[6:6 + 2 * datatypes])
        e.dat_offsets = dat_offsets

        # If config data is missing, set
        if not cls.config:
            cls.config = Config.from_bytes(
                bytes[dat_offsets[0]:dat_offsets[0] + cls.config_length])

        # Decode the rest of the data
        # Datatypes through voltage
        # Offset used to make indexing significantly easier
        offset = dat_offsets[1]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        e.number = struct.unpack("H", bytes[offset:offset+2])[0]
        offset += 2

        rtc = struct.unpack("7B", bytes[offset:offset + 7])
        e.mtime = np.float64(datetime(rtc[0] + 2000, rtc[1], rtc[2],
                                      rtc[3], rtc[4], rtc[5]).timestamp())
        offset += 12  # RTC length + 5 unused bytes

        e.depth = np.float32(struct.unpack(
            "H", bytes[offset:offset+2])[0] * 0.1)
        offset += 8  # depth length + 6 unused bytes

        e.salinity = np.int16(struct.unpack("h", bytes[offset:offset+2])[0])
        offset += 2

        e.temperature = np.float32(struct.unpack(
            "h", bytes[offset:offset+2])[0] * 0.01)
        offset += 2

        sleep = struct.unpack("3B", bytes[offset:offset+3])
        # Converts to seconds
        e.mpt = np.sum(np.multiply(sleep, (60, 1, 0.01)))
        offset += 12  # sleep length + 9 unused bytes

        e.voltage = np.float32(bytes[offset] * 157 / 1000)

        # Velocity data
        n_cells = cls.config.n_cells
        offset = dat_offsets[2]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        vels = np.frombuffer(bytes, dtype=np.int16, count=4 *
                             n_cells, offset=offset).reshape((n_cells, 4)) * 0.001
        e.x_vel = vels[:, 0].copy()
        e.y_vel = vels[:, 1].copy()
        e.z_vel = vels[:, 2].copy()
        offset += 4 * n_cells

        # Correlation data
        offset = dat_offsets[4]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        corr = np.frombuffer(bytes, dtype=np.uint8, count=4 *
                             n_cells, offset=offset).reshape((n_cells, 4))
        e.corr = corr[:, 0:3]

        # Intensity data
        offset = dat_offsets[5]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        intens = np.frombuffer(
            bytes, dtype=np.uint8, count=4 * n_cells, offset=offset).reshape((n_cells, 4))
        e.intens = intens[:, 0:3]

        # Percent good data
        offset = dat_offsets[6]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        perc_good = np.frombuffer(
            bytes, dtype=np.uint8, count=4 * n_cells, offset=offset).reshape((n_cells, 4))
        e.perc_good = perc_good[:, 0:3]

        # Surface track data
        offset = dat_offsets[10]
        # cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        offset += 2

        e.surface_track = np.float32(struct.unpack(
            "I", bytes[offset:offset + 4])[0] * 0.0001)
        offset += 4

        e.surface_track_uncorr = np.float32(struct.unpack(
            "I", bytes[offset:offset + 4])[0] * 0.0001)
        offset += 5  # ununcorrected surface length + 1 unused byte

        e.v_amp = np.uint8(bytes[offset])
        offset += 1

        e.v_pgood = np.uint8(bytes[offset])

        return e


# TODO: Add these whenever checking cfgid to make sure data headers match up
class EnsembleFormatError(Exception):
    """Raised when ensemble configuration is missing or invalid, other formatting errors"""
    pass


decode_bin(path)
