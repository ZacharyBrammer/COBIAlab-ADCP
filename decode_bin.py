import argparse
import os
import struct
from dataclasses import dataclass, field
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

    print(ens_indexes)

    file.seek(ens_indexes[0], 0)
    # print(file.tell())
    cfgid = file.read(2)
    # print(cfgid)
    # print(struct.unpack("2B", cfgid))
    numbytes = file.read(2)
    # print(numbytes)
    realnumbytes = struct.unpack("<h", numbytes)
    # print(realnumbytes)
    # print(file.tell())
    file.seek(1, 1)
    datatypes = file.read(1)
    # print(datatypes)
    # print(struct.unpack("b", datatypes))
    num_data_types = struct.unpack("b", datatypes)[0]
    # print(file.tell())
    dat_offsets = file.read(2 * num_data_types)
    # print(file.tell())
    # print(struct.unpack(f"<{num_data_types}h", dat_offsets))
    # INDEXES FOR SENDING TO CLASS
    ens_start_end = (ens_indexes[0], ens_indexes[0] + realnumbytes[0])
    # print(ens_start_end)

    file.seek(ens_indexes[0], 0)
    ens_dat = file.read(realnumbytes[0])
    # print(len(ens_dat))
    # print(file.tell())

    ens = Ensemble.from_bytes(ens_dat)

    """
    # Set up structure of HDF5 file
    with h5py.File(f"{path}.hdf5", "w") as f:
        pass
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
    number: int = 0
    mtime: float = 0.0
    depth: float = 0.0
    salinity: int = 0
    temperature: float = 0.0
    mpt: float = 0.0
    voltage: float = 0.0
    x_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))
    corr: np.ndarray = field(default_factory=lambda: np.zeros(1))
    intens: np.ndarray = field(default_factory=lambda: np.zeros(1))
    perc_good: np.ndarray = field(default_factory=lambda: np.zeros(1))
    surface_track: float = 0.0
    surface_track_uncorr: float = 0.0
    v_amp: int = 0
    v_pgood: int = 0

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
        e.mtime = datetime(rtc[0] + 2000, rtc[1], rtc[2],
                           rtc[3], rtc[4], rtc[5]).timestamp()
        offset += 12  # RTC length + 5 unused bytes

        e.depth = struct.unpack("H", bytes[offset:offset+2])[0] * 0.1
        offset += 8  # depth length + 6 unused bytes

        e.salinity = struct.unpack("h", bytes[offset:offset+2])[0]
        offset += 2

        e.temperature = struct.unpack("h", bytes[offset:offset+2])[0] * 0.01
        offset += 2

        sleep = struct.unpack("3B", bytes[offset:offset+3])
        # Converts to seconds
        e.mpt = np.sum(np.multiply(sleep, (60, 1, 0.01)))
        offset += 12  # sleep length + 9 unused bytes

        e.voltage = bytes[offset] * 157 / 1000

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

        e.surface_track = struct.unpack(
            "I", bytes[offset:offset + 4])[0] * 0.0001
        offset += 4

        e.surface_track_uncorr = struct.unpack(
            "I", bytes[offset:offset + 4])[0] * 0.0001
        offset += 5  # ununcorrected surface length + 1 unused byte

        e.v_amp = bytes[offset]
        offset += 1

        e.v_pgood = bytes[offset]

        return e


decode_bin(path)
