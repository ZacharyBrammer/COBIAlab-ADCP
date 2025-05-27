import argparse
import os

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

# Converrt file to readable data
def decode_bin(path):
    file = open(path, "rb")

    # Get first data batch
    bufferSize = int(1e6)
    eoe = 127

    # Get the end of the file (needed for when file is larger than buffer size)
    file.seek(0, 2)
    eof = file.tell()
    file.seek(0, 0) # Reset position

    # Read a batch of data
    batch = np.fromfile(file, dtype="uint8", count=bufferSize)

    # Appending with numpy arrays is inefficient so use a list
    increment = list(np.fromfile(file, dtype="uint8", count=2))

    # Search until the next ensemble starts so nothing gets split
    while (len(increment) < 2 or (increment[-1] != eoe and increment[-2] != eoe)) and (file.tell() != eof):
        # Read the next byte
        nextByte = file.read(1)

        # If byte is empty, break from loop
        if not nextByte:
            break
        increment.append(nextByte[0])
    
    # Convert increment to ndarray and add to data
    increment = np.array(increment, dtype="uint8")
    data = np.concatenate((batch, increment[:-1]))

    
    

decode_bin(path)
