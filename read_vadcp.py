# Read VADCP data from serial port and save to file

import datetime
import time

import serial

while True:
    try:
        # open data incoming serial port
        ser1in = serial.Serial('/dev/ttyUSB0', 115200)
        print("Opening vadcp serial port: ", ser1in.name)

        datafile = '/home/cobialab/measurements/measurements.000'
        vadcp1 = open(datafile, 'ab')  # open file for writing

        read_byte1 = ser1in.read()
        count = 0
        while read_byte1 is not None:  # loop over serial port
            read_byte1 = ser1in.read()  # read data from serial port
            vadcp1.write(read_byte1)  # write data to file
            vadcp1.close()  # close file
            vadcp1 = open(datafile, 'ab')  # reopen file
            print("Data written at :", str(
                datetime.datetime.now()), "to ", str(vadcp1.name))

    except serial.SerialException as e:
        print(e)
        time.sleep(5)
