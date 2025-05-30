import argparse
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

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
    config: Config | None = None
    config_length = 58  # Fixed length from operation manual

    def __init__(self):
        self.datatypes = None
        self.dat_offsets = None
        self.number = None
        self.mtime = None
        self.depth = None
        self.salinity = None
        self.temperature = None
        self.mpt = None
        self.voltage = None
        self.x_vel = None
        self.y_vel = None
        self.z_vel = None
        self.corr = None
        self.intens = None
        self.perc_good = None
        self.surface_track = None
        self.surface_track_uncorr = None
        self.v_amp = None
        self.v_pgood = None

    @classmethod
    def from_bytes(cls, bytes):
        e = cls()

        # Read header
        print(len(bytes))
        cfgid = bytes[:2]
        print(cfgid)
        print(struct.unpack("2B", cfgid))
        numbytes = bytes[2:4]
        realnumbytes = struct.unpack("<h", numbytes)
        print(realnumbytes)
        datatypes = struct.unpack("b", bytes[5:6])[0]
        e.datatypes = datatypes
        print(e.datatypes)

        dat_offsets = struct.unpack(
            f"<{datatypes}h", bytes[6:6 + 2 * datatypes])
        e.dat_offsets = dat_offsets
        print(dat_offsets)

        # If config data is missing, set
        if not cls.config:
            print(dat_offsets[0] + cls.config_length)
            cls.config = Config.from_bytes(
                bytes[dat_offsets[0]:dat_offsets[0] + cls.config_length])

        # Decode the rest of the data

        return e


decode_bin(path)
