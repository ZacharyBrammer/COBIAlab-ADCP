import calendar
import struct
from dataclasses import dataclass, field
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
        if bytes[0] != 0 or bytes[1] != 0:
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


@dataclass(slots=True)
class Ensemble:
    """Class for storing and decoding ensemble data"""
    config: ClassVar[Optional[Config]] = None
    config_length: ClassVar[int] = 58  # Fixed length from operation manual
    datatypes: int = 0
    dat_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int16))
    number: np.uint32 = np.uint32(0)
    mtime: np.uint32 = np.uint32(0)
    depth: np.uint16 = np.uint16(0)
    salinity: np.int16 = np.int16(0)
    temperature: np.int16 = np.int16(0)
    mpt: np.uint32 = np.uint32(0)
    voltage: np.uint8 = np.uint8(0)
    x_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int16))
    y_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int16))
    z_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.int16))
    corr: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    intens: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    perc_good: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.uint8))
    surface_track: np.uint32 = np.uint32(0)
    surface_track_uncorr: np.uint32 = np.uint32(0)
    v_amp: np.uint8 = np.uint8(0)
    v_pgood: np.uint8 = np.uint8(0)

    @classmethod
    def from_bytes(cls, bytes):
        e = cls()

        # Check that the header matches
        if bytes[0] != 127 or bytes[1] != 127:
            raise EnsembleFormatError("Header not at expected index")

        # numbytes = bytes[2:4]
        datatypes = bytes[5]
        e.datatypes = datatypes

        dat_offsets = np.frombuffer(bytes, np.int16, count=datatypes, offset=6)
        e.dat_offsets = dat_offsets

        # If config data is missing, set
        if not cls.config:
            cls.config = Config.from_bytes(
                bytes[dat_offsets[0]:dat_offsets[0] + cls.config_length])

        # Decode the rest of the data
        # Datatypes through voltage
        # Offset used to make indexing significantly easier
        offset = dat_offsets[1]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 128 or bytes[offset+1] != 0:
            raise EnsembleFormatError("Variable leader not at expected index")
        offset += 2

        # Number is made up of bytes 3 and 4, with byte 12 for rollover
        number = np.uint16(struct.unpack_from('<H', bytes, offset)[0])
        # Bitwise shift is faster and easier than multiplying and adding
        e.number = np.uint32((bytes[11] << 16) | number)
        offset += 2

        # Real time clock (Bytes 5-11)
        year, mon, day, hour, minute, second, _ = struct.unpack_from('<7B', bytes, offset)
        rtc_tuple = (year + 2000, mon, day, hour, minute, second)
        e.mtime = np.uint32(calendar.timegm(rtc_tuple))
        offset += 12  # RTC length + 5 unused bytes

        # Depth (Bytes 17, 18)
        e.depth = np.uint16(struct.unpack_from('<H', bytes, offset)[0])
        offset += 8  # depth length + 6 unused bytes

        # Salinity (Bytes 25, 26) and Temperature (Bytes 27, 28)
        sal, temp = struct.unpack_from('<hh', bytes, offset)
        e.salinity = np.int16(sal)
        e.temperature = np.int16(temp)
        offset += 4

        # Sleep (Bytes 29-31)
        sleep = bytes[offset:offset+3]
        # Converts to centi-seconds (hundredths are smallest value in data)
        e.mpt = np.uint32(sleep[0] * 6000 + sleep[1] * 100 + sleep[2])
        offset += 12  # sleep length + 9 unused bytes

        # Battery voltage (Byte 41)
        # Scaling attached to file to save storage space
        e.voltage = bytes[offset]

        # Velocity data
        n_cells = cls.config.n_cells
        offset = dat_offsets[2]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 0 or bytes[offset+1] != 1:
            raise EnsembleFormatError("Velocity ID not at expected index")
        offset += 2

        # Velocity data (Bytes 3-4 * n_cells)
        vels = np.frombuffer(bytes, dtype=np.int16, count=4 * n_cells, offset=offset)
        e.x_vel = vels[0::4]
        e.y_vel = vels[1::4]
        e.z_vel = vels[2::4]

        # Correlation data
        offset = dat_offsets[4]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 0 or bytes[offset+1] != 2:
            raise EnsembleFormatError("Correlation ID not at expected index")
        offset += 2

        # Correlation data (Bytes 3-4 * n_cells)
        corr_flat = np.frombuffer(bytes, dtype=np.uint8, count=4 * n_cells, offset=offset)
        corr = np.empty((n_cells, 3), dtype=np.uint8)
        corr[:, 0] = corr_flat[0::4]
        corr[:, 1] = corr_flat[1::4]
        corr[:, 2] = corr_flat[2::4]
        e.corr = corr

        # Intensity data
        offset = dat_offsets[5]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 0 or bytes[offset+1] != 3:
            raise EnsembleFormatError("Echo ID not at expected index")
        offset += 2

        # Intensity data (Bytes 3-4 * n_cells)
        intens_flat = np.frombuffer(bytes, dtype=np.uint8, count=4 * n_cells, offset=offset)
        intens = np.empty((n_cells, 3), dtype=np.uint8)
        intens[:, 0] = intens_flat[0::4]
        intens[:, 1] = intens_flat[1::4]
        intens[:, 2] = intens_flat[2::4]
        e.intens = intens

        # Percent good data
        offset = dat_offsets[6]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 0 or bytes[offset+1] != 4:
            raise EnsembleFormatError("Percent good ID not at expected index")
        offset += 2

        # Percent good data (Bytes 3-4 * n_cells)
        perc_good_flat = np.frombuffer(bytes, dtype=np.uint8, count=4 * n_cells, offset=offset)
        perc_good = np.empty((n_cells, 3), dtype=np.uint8)
        perc_good[:, 0] = perc_good_flat[0::4]
        perc_good[:, 1] = perc_good_flat[1::4]
        perc_good[:, 2] = perc_good_flat[2::4]
        e.perc_good = perc_good

        # Surface track data
        offset = dat_offsets[10]

        # Check ID (Bytes 1, 2)
        if bytes[offset] != 0 or bytes[offset+1] != 64:
            raise EnsembleFormatError("Surface track ID not at expected index")
        offset += 2

        # Surface track data
        surface_track = struct.unpack_from('<II', bytes, offset)
        offset += 9  # Surface track lengths + 1 unused byte

        # Surface track (Bytes 3-6)
        e.surface_track = surface_track[0]

        # Uncorrected surface track (Bytes 7-10)
        e.surface_track_uncorr = surface_track[1]

        # Amplitude at surface (Byte 12)
        e.v_amp = np.uint8(bytes[offset])
        offset += 1

        # Percent good surface track (Byte 13)
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
            "number": np.empty(batch_size, dtype=np.uint32),
            "mtime": np.empty(batch_size, dtype=np.uint32),
            "depth": np.empty(batch_size, dtype=np.uint16),
            "salinity": np.empty(batch_size, dtype=np.int16),
            "temperature": np.empty(batch_size, dtype=np.int16),
            "mpt": np.empty(batch_size, dtype=np.uint32),
            "voltage": np.empty(batch_size, dtype=np.uint8),
            "x_vel": np.empty((batch_size, n_cells), dtype=np.int16),
            "y_vel": np.empty((batch_size, n_cells), dtype=np.int16),
            "z_vel": np.empty((batch_size, n_cells), dtype=np.int16),
            "corr": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "intens": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "perc_good": np.empty((batch_size, n_cells, 3), dtype=np.uint8),
            "surface_track": np.empty(batch_size, dtype=np.uint32),
            "surface_track_uncorr": np.empty(batch_size, dtype=np.uint32),
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


class EnsembleFormatError(Exception):
    """Raised when ensemble configuration is missing or invalid, other formatting errors"""
    pass
