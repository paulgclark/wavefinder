#!/usr/bin/env python
# This file processes an input iq file and outputs the "location"
# of 
# 
# Needs python 2.7 (installed on almost any possible Linux system)
# bitstring module ("sudo pip install bitstring" on Ubuntu)
import sys
import io
import os
import argparse
import numpy
from gnuradio import blocks
from gnuradio import gr
from gnuradio import digital
from gnuradio import filter
from gnuradio.fft import logpwrfft
from math import pi


# virtual enums
SIGNAL_UNDEFINED = -1
SIGNAL_OOK = 0
SIGNAL_FSK = 1
fftFileName = "tmp_wave_finder_file.fft"
freqTolerance = 3 # pass this?

signalList=[]
class signalInfo():
    def __init__(self, timeStamp, duration, frequency, bandwidth, signalType):
        self.timeStamp = timeStamp
        self.duration = duration
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.signalType = signalType

    def stringVal(self):
        if self.signalType == SIGNAL_OOK:
            self.outString = "OOK @"
        elif self.signalType == SIGNAL_FSK:
            self.outString = "FSK @"
        else:
            self.outString = "??? @"
        self.outString += '{:.3f}'.format(self.timeStamp) + "s, "
        self.outString += '{:.2f}'.format(self.duration/1000.0) + "ms in duration; freq="
        self.outString += '{:4.4f}'.format(self.frequency/1000000) + "MHz, est. bandwidth="
        self.outString += '{:2.2f}'.format(self.bandwidth/1000.0) + "kHz"
        return (self.outString)

class fft_flowgraph(gr.top_block):
    def __init__(self, samp_rate, fft_size, frame_rate,
                 iqFileName, fftFileName):
        # boilerplate
        gr.top_block.__init__(self)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.fft_size = fft_size
        self.frame_rate = frame_rate

        ##################################################
        # Blocks
        ##################################################
        self.blocks_file_source_0 = blocks.file_source(
                gr.sizeof_gr_complex*1, 
                iqFileName, 
                False)
        self.logpwrfft_x_0 = logpwrfft.logpwrfft_c(
                sample_rate=samp_rate,
                fft_size=fft_size,
                ref_scale=2,
                frame_rate=frame_rate,
                avg_alpha=1.0,
                average=False)
        self.blocks_file_sink_0 = blocks.file_sink(
                gr.sizeof_float*fft_size, 
                fftFileName,
                False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.logpwrfft_x_0, 0))    
        self.connect((self.logpwrfft_x_0, 0), (self.blocks_file_sink_0, 0))

#####################################
# preset some command line args
protocol_number = -1 # denotes no protocol given via command line

# handling command line arguments using argparse
parser = argparse.ArgumentParser("Process input I-Q data files and output the time stamps and frequencies of any signals found")
parser.add_argument("-q", "--iq", help="input i-q data file name")
parser.add_argument("-s", "--samp_rate", help="sample rate (kHz)", type=int)
parser.add_argument("-c", "--center_freq", help="center frequency (kHz)",
                    type=int)
parser.add_argument("-z", "--fft_size", help="FFT size, must be power of 2", type=int)
parser.add_argument("-f", "--frame_rate", help="FFT frame rate (frames per sec)", type=int)
parser.add_argument("-n", "--min_snr", help="Minimum SNR (dB)", type=int)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()

# assign args to variables
verbose = args.verbose
if args.iq:
    iqFileName = args.iq
else:
    print "Fatal Error: No IQ file provided"
    exit(0)
if args.fft_size > 0:
    fft_size = args.fft_size
else:
    print "Using default FFT size of 1024"
    fft_size = 1024
if args.samp_rate > 0:
    samp_rate = args.samp_rate * 1000.0 # ensure this is a float
else:
    print "Fatal Error: No sample rate given (or less than zero)"
    exit(0)
if args.center_freq > 0:
    center_freq = args.center_freq * 1000.0 # ensure this is a float
else:
    print "Fatal Error: No center frequency given (or less than zero)"
    exit(0)
if args.frame_rate > 0:
    frame_rate = args.frame_rate
else:
    print "Using default FFT frame rate of 30fps"
    frame_rate = 30
if args.min_snr > 0:
    min_snr = args.min_snr
else:
    print "Using default SNR of 30dB"
    min_snr = 30

# compute FFT for IQ file
if iqFileName:
    try:
        if verbose:
            print "\nRunning Wave Finder..."
            print "IQ File: " + iqFileName
            print "Sample Rate (Hz): " + str(samp_rate)
            print "Center Frequency (Hz): " + str(center_freq)
            print "FFT Size: " + str(fft_size)
            print "FFT Frame Rate: " + str(frame_rate)
        flowgraphObject = fft_flowgraph(samp_rate,
                                        fft_size,
                                        frame_rate,
                                        iqFileName,
                                        fftFileName)
        flowgraphObject.run()
    except [[KeyboardInterrupt]]:
        pass

# read FFT file into a two-dimensional array (list of lists that are fft_size long)
# fftFloat     - list of the floats obtained from the binary file
# fftFrameList - list of lists, with each sub-list fft_size in length
try:
    fftFloat = numpy.fromfile(fftFileName, dtype=numpy.float32)
    numFrames = len(fftFloat)/fft_size
    fftFrameList = numpy.reshape(fftFloat, (numFrames, fft_size))
    if verbose:
        print "iq File duration = " + str(1.0*numFrames/frame_rate) + "sec"
except:
    print "Error: FFT file missing"

# displays FFT file properties
#fileInfo = os.stat(fftFileName)
#print fileInfo.st_size
#print len(fftFloat)
#print len(fftFrameList)

# search through FFT file to find any "sudden" changes in signal level
noiseLevel = -40.0 # default value
# take the mean of the mean of each frame to establish the noise level
meanList = []
for frame in fftFrameList:
    meanList.append(numpy.mean(frame))
noiseLevel = numpy.mean(meanList)
signalThresh = noiseLevel + min_snr
if verbose:
    print "Noise Level = " + str(noiseLevel) + "  Min Signal = " + str(signalThresh)
        
frameCount = 0
signalPointList = [] # contains coordinates (time, freq) of all detected maxima
for frame in fftFrameList:
    # see if any buckets in the frame exceed the noise level, omitting any
    # maxima that occurs at time = 0
    if (max(frame) > signalThresh) and (frameCount != 0):
        maxBucketIndex = numpy.argmax(frame)
        signalPointList.append((frameCount, maxBucketIndex))
        # to handle multiple transmissions and FSK, will need to find local maxima
    frameCount += 1

if verbose:
    print signalPointList
    
# merge any adjacent occurrences of signal energy
signalDurationList = []
i = 0
while i < len(signalPointList):
    if signalPointList[i][0] >= 0: # ignore all the removed items
        # this will be the starting point for a new transmission
        # it will have a size of 1 or more frames
        currentSignalStart = signalPointList[i][0]
        currentSignalFreq = signalPointList[i][1]
        currentDuration = 1
        lastTime = currentSignalStart
        lastFreq = currentSignalFreq
        # remove this item from further consideration 
        signalPointList[i] = (-10, -10)
        # scan through remaining list for a point contiguous
        j = i + 1 # start at the next item
        while j < len(signalPointList):
            # if this new point is contiguous in time and freq
            newTime = signalPointList[j][0]
            newFreq = signalPointList[j][1]
            if (newTime == (lastTime + 1)) and (abs(currentSignalFreq - newFreq) <= freqTolerance):
                lastTime = signalPointList[j][0]
                lastFreq = signalPointList[j][1] # can we get drifting issues here? use threshold?
                currentDuration += 1
                signalPointList[j] = (-10, -10) # remove this point from further consideration
            j += 1
        # add final signal properties to new list
        signalDurationList.append((currentSignalStart, currentDuration, currentSignalFreq))
    i += 1

print signalDurationList
print "Total Number of Transmissions Found: " + str(len(signalDurationList))
for s in signalDurationList:
    timeStamp = 1.0*s[0]/frame_rate # in seconds
    duration = 1000*s[1]/frame_rate # in milliseconds
    # FFT data runs from zero to half the fs/2, then -fs/2 back down to zero
    if s[2] < fft_size/2:
        frequency = center_freq + 1.0*s[2]*(samp_rate/fft_size)
    else:
        frequency = center_freq - (fft_size - s[2])*(samp_rate/fft_size)
    signal = signalInfo(timeStamp, duration, frequency, 0, SIGNAL_OOK)
    signalList.append(signal)

# print results to sdtout
for signal in signalList:
    print signal.stringVal()
