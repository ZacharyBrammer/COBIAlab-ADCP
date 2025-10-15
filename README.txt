This repository contains both the code for collecting ADCP data on the Raspberry Pis, along with decoding the output data files.
To decode the binary files, first run "pip install requirements.txt". After this is done, to decode any file run "python decode_bin.py path_to_binary.000".
The resulting HDF5 file will be output in the same directory as the binary file with the same name, but it will have the HDF5 extension.
These files can be easily viewed using either Matlab or Python, depending on personal preference.
