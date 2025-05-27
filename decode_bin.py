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

# Convert file to readable data
def decode_bin(path):
    file = open(path, "rb")

    # Get first data batch
    buffer_size = int(1e6)
    eoe = 127

    # Get the end of the file (needed for when file is larger than buffer size)
    file.seek(0, 2)
    eof = file.tell()
    file.seek(0, 0) # Reset position

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
    

decode_bin(path)
