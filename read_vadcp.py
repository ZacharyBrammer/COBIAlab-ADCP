#!/usr/pbin/python
#Read VADCP data from serial port and save to file

import serial
import datetime

#print "Data written at :", str(datetime.datetime.now())

ser1in = serial.Serial('/dev/ttyUSB0',115200) #open data incoming serial port
print "Opening vadcp serial port: ", ser1in.name #check correct serial port was opened

#ser1out = serial.Serial('/dev/ttyUSB1',4800) #open send data port
#print "Opening send data serial port: ", ser1out.name

datafile='/home/cobialab/measurements/measurements'

vadcp1 = open(datafile,'a') #open file for writing

read_byte1 = ser1in.read() 
count = 0
while read_byte1 is not None:  #loop over serial port
	read_byte1 = ser1in.read() #read data from serial port
	vadcp1.write(read_byte1) #write data to file
	vadcp1.close() #close file
	vadcp1 = open(datafile,'a') #reopen file
	print "Data written at :", str(datetime.datetime.now()),"to ",str(vadcp1.name)
	#ser1out.write(read_byte1)

