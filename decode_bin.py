import argparse
import os

# Command line argument for file path
parser = argparse.ArgumentParser(
    prog="DecodeADCP",
    description="Parses VADCP output data and decodes it"
)

parser.add_argument(dest="path", help="Path to data file")
args = parser.parse_args()

path = args.path
print(path)
if os.path.exists(path):
    pass
else:
    raise FileNotFoundError("Invalid Path Provided")

file = open(path, "rb")
