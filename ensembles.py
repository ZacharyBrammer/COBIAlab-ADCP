import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, cast

import h5py
import numpy as np


@dataclass
class Config:
    """Class for storing and decoding config data"""
    # Since there's only one config native types are fine - don't have to worry about saving memory
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

        # Check ID
        cfgid = struct.unpack("2B", bytes[:2])
        if cfgid != (0, 0):
            raise EnsembleFormatError("Config ID not at expected index")

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
        c.bandwidth = struct.unpack("H", bytes[50:52])[0] * 0.01
        c.syspower = bytes[52]
        c.sernum = struct.unpack("I", bytes[54:58])[0]
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
        cfgid = struct.unpack("2B", bytes[:2])

        # Check that the header matches
        if cfgid != (127, 127):
            raise EnsembleFormatError("Header not at expected index")
        
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

        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (128, 0):
            raise EnsembleFormatError("Variable leader not at expected index")
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

        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (0, 1):
            raise EnsembleFormatError("Velocity ID not at expected index")
        offset += 2

        vels = np.frombuffer(bytes, dtype=np.int16, count=4 *
                             n_cells, offset=offset).reshape((n_cells, 4)) * 0.001
        e.x_vel = vels[:, 0].copy()
        e.y_vel = vels[:, 1].copy()
        e.z_vel = vels[:, 2].copy()
        offset += 4 * n_cells

        # Correlation data
        offset = dat_offsets[4]
        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (0, 2):
            raise EnsembleFormatError("Correlation ID not at expected index")
        offset += 2

        corr = np.frombuffer(bytes, dtype=np.uint8, count=4 *
                             n_cells, offset=offset).reshape((n_cells, 4))
        e.corr = corr[:, 0:3]

        # Intensity data
        offset = dat_offsets[5]
        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (0, 3):
            raise EnsembleFormatError("Echo intensity ID not at expected index")
        offset += 2

        intens = np.frombuffer(
            bytes, dtype=np.uint8, count=4 * n_cells, offset=offset).reshape((n_cells, 4))
        e.intens = intens[:, 0:3]

        # Percent good data
        offset = dat_offsets[6]
        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (0, 4):
            raise EnsembleFormatError("Percent good ID not at expected index")
        offset += 2

        perc_good = np.frombuffer(
            bytes, dtype=np.uint8, count=4 * n_cells, offset=offset).reshape((n_cells, 4))
        e.perc_good = perc_good[:, 0:3]

        # Surface track data
        offset = dat_offsets[10]
        # Check ID
        cfgid = struct.unpack("2B", bytes[offset:offset + 2])
        if cfgid != (0, 64):
            raise EnsembleFormatError("Surface track ID not at expected index")
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


class EnsembleWriter:
    """Class for writing a batch of ensembles to an hdf5 file"""

    def __init__(self, filename: str, batch_size: int, n_cells: int):
        self.filename: str = filename
        self.batch_size: int = batch_size
        self.n_cells: int = n_cells

        # Set up arrays in advance to save memory (not having to make new arrays each write)
        self.arrays: Dict[str, np.ndarray] = {
            "number": np.empty(batch_size, dtype=np.uint16),
            "mtime": np.empty(batch_size, dtype=np.float64),
            "depth": np.empty(batch_size, dtype=np.float32),
            "salinity": np.empty(batch_size, dtype=np.int16),
            "temperature": np.empty(batch_size, dtype=np.float32),
            "mpt": np.empty(batch_size, dtype=np.float64),
            "voltage": np.empty(batch_size, dtype=np.float32),
            "x_vel": np.empty((batch_size, n_cells), dtype=np.float64),
            "y_vel": np.empty((batch_size, n_cells), dtype=np.float64),
            "z_vel": np.empty((batch_size, n_cells), dtype=np.float64),
            "corr": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "intens": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "perc_good": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "surface_track": np.empty(batch_size, dtype=np.float32),
            "surface_track_uncorr": np.empty(batch_size, dtype=np.float32),
            "v_amp": np.empty(batch_size, dtype=np.uint8),
            "v_pgood": np.empty(batch_size, dtype=np.uint8),
        }

    def fill_arrays_from_batch(self, batch: List[Ensemble]):
        for i, e in enumerate(batch):
            self.arrays["number"][i] = e.number
            self.arrays["mtime"][i] = e.mtime
            self.arrays["depth"][i] = e.depth
            self.arrays["salinity"][i] = e.salinity
            self.arrays["temperature"][i] = e.temperature
            self.arrays["mpt"][i] = e.mpt
            self.arrays["voltage"][i] = e.voltage

            self.arrays["x_vel"][i, :] = e.x_vel
            self.arrays["y_vel"][i, :] = e.y_vel
            self.arrays["z_vel"][i, :] = e.z_vel

            self.arrays["corr"][i, :, :] = e.corr
            self.arrays["intens"][i, :, :] = e.intens
            self.arrays["perc_good"][i, :, :] = e.perc_good

            self.arrays["surface_track"][i] = e.surface_track
            self.arrays["surface_track_uncorr"][i] = e.surface_track_uncorr
            self.arrays["v_amp"][i] = e.v_amp
            self.arrays["v_pgood"][i] = e.v_pgood

    def write_batch(self, batch: List[Ensemble]):
        batch_len = len(batch)
        if batch_len == 0:
            return

        for ens in batch:
            print(ens.number)

        # Fill buffers with batch data
        self.fill_arrays_from_batch(batch)

        with h5py.File(self.filename, "a") as f:
            # Had to do cast stuff to get type checking to play nicely
            ens_group = cast(h5py.Group, f["ensembles"])

            # Current size of datasets (all should be same length)
            number_ds = cast(h5py.Dataset, ens_group["number"])
            current_size = number_ds.shape[0]
            new_size = current_size + batch_len

            # Resize datasets
            for name in ens_group:
                ds = cast(h5py.Dataset, ens_group[name])
                shape = list(ds.shape)
                shape[0] = new_size
                ds.resize(tuple(shape))

            # Write data slice
            for name, arr in self.arrays.items():
                ds = cast(h5py.Dataset, ens_group[name])
                if arr.ndim == 1:
                    ds[current_size:new_size] = arr[:batch_len]
                else:
                    ds[current_size:new_size, ...] = arr[:batch_len]


# TODO: Add these whenever checking cfgid to make sure data headers match up
class EnsembleFormatError(Exception):
    """Raised when ensemble configuration is missing or invalid, other formatting errors"""
    pass
