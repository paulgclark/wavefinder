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
from gnuradio import gr
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio.fft import logpwrfft
from math import pi

# TURN THIS ON TO SEE ENHANCED DEBUG INFO
debug = False

# virtual enums
SIGNAL_UNDEFINED = -1
SIGNAL_OOK = 0
SIGNAL_FSK = 1
fftFileName = "tmp_wave_finder_file.fft"
freqTolerance = 3 # pass this?
bytesPerSamp = 8
chunkSize = 1024
defaultBuf = 0.05
# there appears to be a problem with gnuradio's FFT block when
# the frame rate is pushed past 200
defaultFrameRate = 200
defaultSNR = 20
defaultFFTSize = 1024

signalList=[]

def roundDown(num, divisor):
    return num - (num % divisor)

def roundUp(num, divisor):
    return (num + divisor) - ((num + divisor) % divisor)

def fileNameTextToFloat(valStr, unitStr):
    # if there's a 'p' character, then we have to deal with decimal vals
    if 'p' in valStr:
        print "decimal value found"
        regex = re.compile(r"([0-9]+)p([0-9]+)")
        wholeVal = regex.findall(valStr)[0][0]
        decimalVal = regex.findall(valStr)[0][1]
        baseVal = 1.0*int(wholeVal) + 1.0*int(decimalVal)/10**len(decimalVal)
    else:
        baseVal = 1.0*int(valStr)

    if unitStr == "G":
        multiplier = 1e9
    elif unitStr == "M":
        multiplier = 1e6
    elif unitStr == "k":
        multiplier = 1e3
    else:
        multiplier = 1.0

    return baseVal * multiplier


import re
class iqFileObject():
    def __init__(self, prefix = None, centerFreq = None, 
                       sampRate = None, fileName = None):
        # if no file name is specified, store the parameters
        if fileName is None:
            self.prefix = prefix
            self.centerFreq = centerFreq
            self.sampRate = sampRate
        # if the file name is specified, we must derive the parameters
        # from the file name
        else:
            # first check if we have a simple file name or a name+path
            regex = re.compile(r"\/")
            if regex.match(fileName):
                # separate the filename from the rest of the path
                regex = re.compile(r"\/([a-zA-Z0-9_.]+)$")
                justName = regex.findall(fileName)[0]
            else:
                justName = fileName
            # get the substrings representing the values
            regex = re.compile(r"_c([0-9p]+)([GMK])_s([0-9p]+)([GMk])\.iq$")
            paramList = regex.findall(justName)
            centerValStr = paramList[0][0]
            centerUnitStr = paramList[0][1]
            sampValStr = paramList[0][2]
            sampUnitStr = paramList[0][3]

            if debug:
                print centerValStr
                print centerUnitStr
                print sampValStr
                print sampUnitStr

            # compute center frequency and sample rate
            self.centerFreq = fileNameTextToFloat(centerValStr, centerUnitStr)
            self.sampRate = fileNameTextToFloat(sampValStr, sampUnitStr)

            # get the prefix
            nonPrefixLen = len("_c" + centerValStr + centerUnitStr +\
                               "_s" + sampValStr + sampUnitStr + ".iq")
            self.prefix = justName[0:len(justName)-nonPrefixLen]
             
            if debug:
                print self.centerFreq
                print self.sampRate
                print self.prefix

    def fileName(self):
        tempStr = self.prefix
        # add center frequency
        # first determine if we should use k, M, G or nothing
        # then divide by the appropriate unit
        if self.centerFreq > 1e9:
            unitMag = 'G'
            wholeVal = int(1.0*self.centerFreq/1e9)
            decimalVal = (1.0*self.centerFreq - 1e9*wholeVal)
            decimalVal = int(decimalVal/1e7)
        elif self.centerFreq > 1e6:
            unitMag = 'M'
            wholeVal = int(1.0*self.centerFreq/1e6)
            decimalVal = (1.0*self.centerFreq - 1e6*wholeVal)
            decimalVal = int(decimalVal/1e4)
        elif self.centerFreq > 1e3:
            unitMag = 'k'
            wholeVal = int(1.0*self.centerFreq/1e3)
            decimalVal = (1.0*self.centerFreq - 1e3*wholeVal)
            decimalVal = int(decimalVal/1e1)
        else:
            unitMag = ''
            value = int(self.centerFreq)
        if decimalVal == 0:
            tempStr += "_c{}{}".format(wholeVal, unitMag)
        else: 
            tempStr += "_c{}p{}{}".format(wholeVal, decimalVal, unitMag)

        # do the same thing for the sample rate
        if self.sampRate > 1e6:
            unitMag = 'M'
            wholeVal = int(1.0*self.sampRate/1e6)
            decimalVal = (1.0*self.sampRate - 1e6*wholeVal)
            decimalVal = int(decimalVal/1e4)
        elif self.sampRate > 1e3:
            unitMag = 'k'
            wholeVal = int(1.0*self.sampRate/1e3)
            decimalVal = (1.0*self.sampRate - 1e3*wholeVal)
            value = self.sampRate/1e1
        else:
            unitMag = ''
            value = int(self.sampRate)
        if decimalVal == 0:
            tempStr += "_s{}{}".format(wholeVal, unitMag)
        else: 
            tempStr += "_s{}p{}{}".format(wholeVal, decimalVal, unitMag)
        tempStr += ".iq"
        return tempStr
        

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
        self.outString += '{:.2f}'.format(1000.0 * self.duration) + "ms in duration; freq="
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



class decFlowgraph(gr.top_block):
    def __init__(self, inputFileName, outputFileName, 
                 center_freq, center_freq_out,
                 samp_rate, samp_rate_out):
        # boilerplate
        gr.top_block.__init__(self)

        ##################################################
        # Variables
        ##################################################
        cutoff_freq = 1.1*samp_rate_out
        transition_width = 0.1*samp_rate_out
        firdes_taps = firdes.low_pass(1, 
                                      samp_rate, 
                                      cutoff_freq, 
                                      transition_width)
        frequency_shift = center_freq_out - center_freq
        decimation = int(samp_rate/samp_rate_out)
        ##################################################
        # Blocks
        ##################################################
        self.freq_xlating_fir_filter_0 = filter.freq_xlating_fir_filter_ccc(
                                                decimation,
                                                firdes_taps,
                                                frequency_shift,
                                                samp_rate)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, 
                                                       inputFileName,
                                                       False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 
                                                   outputFileName,
                                                   False)
        self.blocks_file_sink_0.set_unbuffered(False)
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), 
                     (self.freq_xlating_fir_filter_0, 0))    
        self.connect((self.freq_xlating_fir_filter_0, 0),
                     (self.blocks_file_sink_0, 0))

#####################################
# preset some command line args
protocol_number = -1 # denotes no protocol given via command line

# handling command line arguments using argparse
parser = argparse.ArgumentParser("Process input I-Q data files and output the time stamps and frequencies of any signals found\n\nIf you tell WaveFinder to PRUNE your file, will will produce a new, minimized file with the supplied prune file name")
parser.add_argument("-q", "--iq", help="input i-q data file name")
parser.add_argument("-s", "--samp_rate", help="sample rate (kHz)", type=int)
parser.add_argument("-c", "--center_freq", help="center frequency (kHz)",
                    type=int)
parser.add_argument("-z", "--fft_size", help="FFT size, must be power of 2", type=int)
parser.add_argument("-f", "--frame_rate", help="FFT frame rate (frames per sec) - PLEASE LEAVE AT DEFAULT", type=int)
parser.add_argument("-n", "--min_snr", help="Minimum SNR (dB)", type=int)
parser.add_argument("-p", "--prune", help="output a minimized IQ file containing only the signals", action="store_true")
parser.add_argument("-o", "--out", help="output pruned i-q file name")
parser.add_argument("-b", "--buf", help="Extra amount to save on prune (ms)", type=int)
parser.add_argument("-m", "--min_duration", help="Ignore signals that don't last at least this long (ms)", type=int)
parser.add_argument("-l", "--list_freqs", help="List the supplied number of most commonly found frequencies", type=int)
parser.add_argument("-d", "--decimate", help="minimize size by decimating",
                    action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()

# assign args to variables
verbose = args.verbose
decimate = args.decimate
prune = args.prune
if args.iq:
    iqFileName = args.iq
    # try to parse the file name to see if if contains the iq parameters
    inputFileObject = iqFileObject(fileName = iqFileName)
    try:
        center_freq = inputFileObject.centerFreq
        samp_rate = inputFileObject.sampRate
    # since the parameters weren't there, they must be supplied from other args
    except:
        center_freq = -1
        samp_rate = -1
    print center_freq
else:
    print "Fatal Error: No IQ file provided"
    exit(0)

# args supplied here override those extracted from the file name
if args.samp_rate > 0:
    samp_rate = args.samp_rate * 1000.0 # ensure this is a float
elif samp_rate < 0:
    print "Fatal Error: No sample rate given (or less than zero)"
    exit(0)
if args.center_freq > 0:
    center_freq = args.center_freq * 1000.0 # ensure this is a float
elif center_freq < 0:
    print "Fatal Error: No center frequency given (or less than zero)"
    exit(0)
if args.out:
    pruneFileName = args.out
else:
    pruneFileName = ""
if args.fft_size > 0:
    fft_size = args.fft_size
else:
    fft_size = defaultFFTSize
    print "Using default FFT size of " + str(defaultFFTSize)
if args.frame_rate > 0:
    if args.frame_rate > 200:
        frame_rate = defaultFrameRate
        print "Maximum FFT Frame Rate is 200fps. Setting frame rate to 200"
    else: 
        frame_rate = args.frame_rate
        if frame_rate != 200:
            print "WARNING: You should use a frame rate of 200fps"
else:
    frame_rate = defaultFrameRate
    #print "Using default FFT frame rate of " + str(defaultFrameRate) + "fps"
if args.min_snr > 0:
    min_snr = args.min_snr
else:
    min_snr = defaultSNR
    print "Using default SNR of " + str(min_snr) + "dB"
if args.buf > 0:
    buf = args.buf/1000.0
else:
    if prune:
        print "Using default buffer size of " + str(defaultBuf) + "ms"
        buf = defaultBuf
if args.min_duration > 0:
    minDuration = args.min_duration/1000.0
else:
    minDuration = 0.0
if args.list_freqs > 0:
    frequencyListCount = args.list_freqs
else:
    frequencyListCount = 0

# compute FFT for IQ file
if iqFileName:
    try:
        if verbose:
            print "\nRunning Wave Finder with cmd line args:"
            print "    IQ File: " + iqFileName
            print "    Sample Rate (Hz): " + str(samp_rate)
            print "    Center Frequency (Hz): " + str(center_freq)
            print "    FFT Size: " + str(fft_size)
            print "    FFT Frame Rate: " + str(frame_rate) + "\n"
        flowgraphObject = fft_flowgraph(samp_rate,
                                        fft_size,
                                        frame_rate,
                                        iqFileName,
                                        fftFileName)
        flowgraphObject.run()
    except [[KeyboardInterrupt]]:
        pass

if verbose:
    print "Flowgraph completed"

# read FFT file into 2-dimensional array (list of lists that are fft_size long)
# fftFloat     - list of the floats obtained from the binary file
# fftFrameList - list of lists, with each sub-list fft_size in length
try:
    fftFloat = numpy.fromfile(fftFileName, dtype=numpy.float32)
    numFrames = len(fftFloat)/fft_size
    fftFrameList = numpy.reshape(fftFloat, (numFrames, fft_size))
    if verbose:
        print "    Number of FFT Frames = " + str(numFrames)
        print "    iq File duration = " + str(1.0*numFrames/frame_rate) + "sec"
except:
    print "Error: FFT file missing"

# displays FFT file properties
if debug:
    fileInfo = os.stat(fftFileName)
    print "FFT File Size: " + str(fileInfo.st_size)
    print "Size of FFT Float list (1-dimensional): " + str(len(fftFloat))
    print "Size of FFT Frame List (2-dimensional): " + str(len(fftFrameList))

# want to ignore the DC spike for the purposes of calulating the noise
# floor as well as finding signal spikes
dcBlockLow = int(0.003*fft_size) # using half a percent
dcBlockHigh = fft_size - dcBlockLow
# print "DC Block indices: " + str(dcBlockLow) + " " + str(dcBlockHigh)

# search through FFT file to find any "sudden" changes in signal level
noiseLevel = -60.0 # default value
# take the mean of the mean of each frame to establish the noise level
meanList = []
for frame in fftFrameList:
    meanList.append(numpy.mean(frame[dcBlockLow:dcBlockHigh]))
noiseLevel = numpy.mean(meanList)
signalThresh = noiseLevel + min_snr
if verbose:
    print "\nNoise Level = " + str(noiseLevel) + "  Min Signal = " + str(signalThresh)

if debug:
    print "Max Value in each FFT Frame"
    for frame in fftFrameList:
        print numpy.max(frame)


frameCount = 0
signalPointList = [] # contains coordinates (time, freq) of all detected maxima
for frame in fftFrameList:
    # see if any buckets in the frame exceed the noise level, omitting any
    # maxima that occurs at time = 0
    if (max(frame[dcBlockLow:dcBlockHigh]) > signalThresh) and (frameCount != 0):
        maxBucketIndex = numpy.argmax(frame[dcBlockLow:dcBlockHigh])
        signalPointList.append((frameCount, maxBucketIndex))
        # to handle multiple transmissions and FSK, will need to find local maxima
    frameCount += 1

# get unique frame values from SignalPointList (first value in pair)
frameFlowList = []
for (frame, freq) in signalPointList:
    frameFlowList.append(frame)
if debug:
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


        # estimate the bandwidth of the signal using frame in middle
        if debug:
            print "Current Frame Start: " + str(currentSignalStart)
            print "Current Duration: " + str(currentDuration)
            print "Midpoint: " + str(int(currentSignalStart+currentDuration/2))
        # NEED to go through all the frames rather than just the middle one
        # if any frame goes past the peak - 20(?) than flag it as part of BW
        frameBW = fftFrameList[int(currentSignalStart+currentDuration/2)] 
        currentBW = 1
        bwThresh = max(frameBW) - 20 # assume a 20dB drop from max
        #bwThresh = frameBW[currentSignalFreq] - 20 # assume 20dB drop from max
        while True:
            if (currentSignalFreq + currentBW) >= fft_size or \
               (currentSignalFreq - currentBW) < 0:
                break
            elif (frameBW[currentSignalFreq-currentBW] < bwThresh) and \
                 (frameBW[currentSignalFreq+currentBW] < bwThresh):
                break
            else:
                currentBW += 1

        # add final signal properties to new list
        signalDurationList.append((currentSignalStart, currentDuration, \
                                   currentSignalFreq, 2*currentBW-1))
    i += 1

# eliminate signals that don't last long enough
tempList = []
if minDuration > 0:
    for s in signalDurationList:
        if s[1] >= minDuration*frame_rate:
            tempList.append(s)
    signalDurationList = tempList

if debug:
    print "List of signal durations:"
    print signalDurationList
       

print "Total File duration = " + str(1.0*numFrames/frame_rate) + "sec"
print "Total Number of Transmissions Found: " + str(len(signalDurationList))
for s in signalDurationList:
    timeStamp = 1.0*s[0]/frame_rate # in seconds
    duration = 1.0*s[1]/frame_rate # in seconds
    # FFT data runs from zero to half the fs/2, then -fs/2 back down to zero
    if s[2] < fft_size/2:
        frequency = center_freq + 1.0*s[2]*(samp_rate/fft_size)
    else:
        frequency = center_freq - (fft_size - s[2])*(samp_rate/fft_size)
    bandwidth = s[3] * samp_rate/fft_size
    signal = signalInfo(timeStamp, duration, frequency, bandwidth, SIGNAL_OOK)
    signalList.append(signal)

# print results to sdtout
for signal in signalList:
    print signal.stringVal()
print "\n"

# print out the most common frequencies found
freqList = []
for s in signalList:
    freqList.append(s.frequency)
from collections import Counter
freqListCounter = Counter(freqList)
print len(freqList)
print "Most common Freq 1: " + str(freqListCounter.most_common(3)[0][0]) + " " + str(freqListCounter.most_common(3)[0][1])
print "Most common Freq 2: " + str(freqListCounter.most_common(3)[1][0]) + " " + str(freqListCounter.most_common(3)[1][1])
print "Most common Freq 3: " + str(freqListCounter.most_common(3)[2][0]) + " " + str(freqListCounter.most_common(3)[2][1])

# if we haven't been told to prune the file, then we're done
if not prune:
    print "Exiting without pruning or decimation..."
    exit(0)

#########################################
# minimize file size using timestamp info
#########################################
# now flow through the file, discarding data unless we are between
# a pair of save points

# open IQ file
try:
    iqFile = open(iqFileName, "rb")
except:
    print "Error: cannot open IQ file: " + iqFileName
    exit(1)
# open output file for pruned IQ version, making up a file name if not provided
if pruneFileName == "":
    pruneFileObj = iqFileObject(prefix = inputFileObject.prefix + "_pruned", 
                                centerFreq = center_freq,
                                sampRate = samp_rate)
    pruneFileName = pruneFileObj.fileName()

try:
    pruneOutFile = open(pruneFileName, "wb")
except:
    print "Error: cannot open output file for writing: " + pruneFileName
    iqFile.close()
    exit(1)

# go through iq file, flowing a frame of iq data at a time
# if the next value in the list; first get a set of frames
# that we want to use to gate the flow
frameSizeInBytes = int(os.path.getsize(iqFileName)/numFrames)
frameSizeInBytes = roundDown(frameSizeInBytes, 8)

# add buffer to list in terms of frame counts
frameSizeInSeconds = 1.0/frame_rate
bufFrameCount = int(buf/frameSizeInSeconds)
if verbose:
    print "System File Size: " + str(os.path.getsize(iqFileName))
    print "Number of Frames in File: " + str(numFrames)
    print "Frame Size in Bytes: " + str(frameSizeInBytes)
    print "Frame in s = " + str(frameSizeInSeconds)
    print "Buf frame count = " + str(bufFrameCount)

# add buffer frames to frame list
bufferedFrameFlowList = []
for frame in frameFlowList:
    for i in range(-1*bufFrameCount, bufFrameCount):
        if (frame - i) >= 0:
            bufferedFrameFlowList.append(frame + i)

# this produces a set of the unique values in the previous list       
frameFlowSet = set(bufferedFrameFlowList)

# for each FFT Frame
for frame in range(numFrames):
    # get the IQ data corresponding to the frame
    try:
        iqFrameData =  iqFile.read(frameSizeInBytes)
        eofFlag = False
    except:
        iqFrameData =  []
        eofFlag = True

    # quit if out of IQ data, else flow data to pruned file if frame
    # has been flagged for flow
    if eofFlag:
        break
    elif (frame + 1) in frameFlowSet:
        if debug:
            print "Writing Frame " + str(frame)
        pruneOutFile.write(iqFrameData)

# close files and exit
iqFile.close()
pruneOutFile.close()

if not decimate:
    print "exiting without decimation..."
    exit(0)

# frequency range is min and max of frequency ranges discovered
for (i, signal) in enumerate(signalList):
    if i == 0:
        minFreq = signal.frequency - signal.bandwidth/2
        maxFreq = signal.frequency + signal.bandwidth/2
    minFreq = min(minFreq, signal.frequency - signal.bandwidth/2)
    maxFreq = max(maxFreq, signal.frequency + signal.bandwidth/2)

newCenterFreq = int((minFreq + maxFreq)/2)
newSampRate = int(1.5 * (maxFreq - minFreq)) # 1.5 means 50% buffer
# round center freq and sample rate up to nearest 10kHz
newCenterFreq = roundUp(newCenterFreq, 10000)
newSampRate = roundUp(newSampRate, 10000)

# build file name
decimatedFileObj = iqFileObject(prefix = inputFileObject.prefix + "_pruned_dec",
                                centerFreq = newCenterFreq,
                                sampRate = newSampRate)
decimatedFileName = decimatedFileObj.fileName()

# run decimation flowgraph with computed parameters
try:
    if verbose:
        print "\nRunning Decimation Flowgraph with the following parameters:"
        print "    Input File: " + pruneFileName
        print "    Output File: " + decimatedFileName
        print "    Sample Rate (Hz): " + str(samp_rate)
        print "    Sample Rate Out (Hz): " + str(newSampRate)
        print "    Center Frequency (Hz): " + str(center_freq)
        print "    Center Frequency Out (Hz): " + str(newCenterFreq) + "\n"

    flowgraphObject2 = decFlowgraph(inputFileName = pruneFileName,
                                    outputFileName = decimatedFileName,
                                    center_freq = center_freq,
                                    center_freq_out = newCenterFreq,
                                    samp_rate = samp_rate,
                                    samp_rate_out = newSampRate)
    flowgraphObject2.run()
except [[KeyboardInterrupt]]:
    pass

    
