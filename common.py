# all imports
import librosa
import librosa.display
from IPython.display import Audio # needed for correct audio player
from IPython.core.display import display
from MaintletTimer import *
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
# this file is specially for the dataset we collected in the mechanical room
from datetime import datetime
from os.path import isfile, join
from os import listdir
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from numpy.fft import fft, ifft
from scipy.fftpack import fft, ifft
plt.style.use('seaborn-poster')
import time 
import psutil
import wave
import gc
from multiprocessing import Pool
import os
import subprocess
import pickle
pid = os.getpid()
tmp = os.system("sudo renice -n -19 -p " + str(pid))
import collections
from datetime import datetime, timedelta
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.dates as mdates
import sys
import matplotlib.animation as animation
import pylab as pl
from IPython import display
import random
from visualizationCommon import *
from sklearn import metrics
import csv
# gc.disable()

class FileStat:
    """
    Save the stats of files in this dataset
    """
    def __init__(self, filepath, sr, sampleCount, channelCount, duration):
        self.filepath = filepath
        self.sr = sr
        self.sampleCount = sampleCount
        self.channelCount = channelCount
        self.duration = duration
    
    def __str__(self):
        return f"""{"Filepath:":<15} {self.filepath}
{"SampleRate:":<15} {self.sr}
{"SampleCount:":<15} {self.sampleCount}
{"ChannelCount:":<15} {self.channelCount}
{"Duration:":<15} {self.duration} Seconds"""
    
class DatasetStat:
    """
    Save the stats of this dataset
    """
    def __init__(self, startDate, endDate, fileCount, datasetDuration):
        self.startDate = startDate
        self.endDate = endDate
        self.fileCount = fileCount
        self.datasetDuration = datasetDuration
    
    def __str__(self):
        return f"""{"startDate:":<15} {self.startDate}
{"endDate:":<15} {self.endDate}
{"fileCount:":<15} {self.fileCount}
{"datasetDuration:":<15} {self.datasetDuration} Seconds"""

def pathToName(filepath):
    """
    Convert filepath to filename
    """
    return filepath.split('/')[-1]
    
def pathToTime(filepath):
    """
    Convert filepath to datetime with granularity of seconds
    """
    filename = pathToName(filepath)
    timeToken = '_'.join(filename.split('.')[0].split('_')[2:-1])
#     dry_run_04_16_2022_00_00_02_054049.wav
    timeDatetime = datetime.strptime(timeToken, "%m_%d_%Y_%H_%M_%S")
    return timeDatetime

def pathToAccurateTime(filepath):
    """
    Convert filepath to datetime with granularity of milliseconds
    """
    filename = pathToName(filepath)
    timeToken = '_'.join(filename.split('.')[0].split('_')[2:])
#     dry_run_04_16_2022_00_00_02_054049.wav
    timeDatetime = datetime.strptime(timeToken, "%m_%d_%Y_%H_%M_%S_%f")
    return timeDatetime

def getTotalWindowCount(fileCount, sampleCountPerFile, windowSize, stepSize):
    return int((fileCount * sampleCountPerFile - windowSize) / stepSize + 1)

class WindowIterator:
    """
    A class for interating audio files in a dataset dir over time windows
    """
    def __init__(self, filepaths, windowSize, stepSize):
        """
        init function
        
        Args:
            filepaths: list of filepaths in the dataset
            windowSize: size of the time window (unit: count of sample. e.g. 48000)
            stepSize: step size for iterating time windows (unit: count of sample. e.g. 240000)
        """
        self.sampleIndex = 0
        self.windowIndex = 0
        self.time = 0
        self.worldTime = []
        self.filepath = ''
        self.filepaths = filepaths
        self.windowSize = windowSize
        self.stepSize = stepSize
        self.handlers = []
        self.remainDataInThisFile = 0
        self.data = []
        self.sr = -1
        self.timeDelta = -1
        self.sampleCountPerFile = -1
        self.fileIndex = 0
        self.fileCount = len(self.filepaths)
        if self.fileCount <= 0:
            raise ValueError("fileCount is incorrect")
    
    def __iter__(self):
        return self
                   
    def __next__(self):
        """
        Return the next time window
        """
        currentWindow = []
        if self.sr == -1:
            # this is the first file
            self.filepath = self.filepaths[self.fileIndex]
            # sr, self.data = wavfile.read(self.filepath)
            self.data, sr = librosa.load(self.filepath, sr=48000, mono=False)
            self.data = self.data.T
            sampleCountPerFile = self.data.shape[0]
            self.sr = sr
            self.timeDelta = self.stepSize / self.sr 
            self.sampleCountPerFile = sampleCountPerFile
            self.remainDataInThisFile = self.sampleCountPerFile
            self.fileIndex += 1
            
        if self.remainDataInThisFile <= self.stepSize:
            # data is not enough in the current file, we need to use a new file
            if self.fileIndex >= self.fileCount:
                # there is no more new files
                raise StopIteration
            
            # process previous file
            currentFileCursor = self.sampleCountPerFile - self.remainDataInThisFile
            dataInPreviousWindow = self.data[currentFileCursor:,:]
            remainWindowSize = self.windowSize - self.remainDataInThisFile
            remainStepSize = self.stepSize - self.remainDataInThisFile
            baseTime1 = pathToTime(self.filepath)
            timeDelta1 = timedelta(seconds=(currentFileCursor / self.sr))

            # start the new file
            self.filepath = self.filepaths[self.fileIndex]
            # sr, self.data = wavfile.read(self.filepath)
            self.data, sr = librosa.load(self.filepath, sr=48000, mono=False)
            self.data = self.data.T
            sampleCountPerFile = self.data.shape[0]
            self.fileIndex += 1

            if self.sr != sr:
                raise ValueError(f"the sampling rate of file {self.filepath} is {sr} not equal to {self.sr}")
                        
            if self.sampleCountPerFile != sampleCountPerFile:
                raise ValueError(f"the sample count per file of file {self.filepath} is {sampleCountPerFile} not equal to {self.sampleCountPerFile}")
                
            # extract the data
            self.remainDataInThisFile = self.sampleCountPerFile
            currentFileCursor = self.sampleCountPerFile - self.remainDataInThisFile
            dataInCurrentWindow = self.data[currentFileCursor:currentFileCursor+remainWindowSize,:]
            if dataInPreviousWindow.shape[0] == 0:
                currentWindow = dataInCurrentWindow
            else:
                # print(dataInPreviousWindow.shape)
                # print(dataInCurrentWindow.shape)
                currentWindow = np.concatenate((dataInPreviousWindow, dataInCurrentWindow), axis=0)
            # update variables
            self.remainDataInThisFile -= remainStepSize
            
            baseTime2 = pathToTime(self.filepath)
            timeDelta2 = timedelta(seconds=((currentFileCursor+remainWindowSize) / self.sr))
            self.worldTime = [baseTime1+timeDelta1, baseTime2+timeDelta2]
            
        elif self.remainDataInThisFile > self.stepSize and self.remainDataInThisFile < self.windowSize:
            # data is not enough in the current file, we need to access the new file
            if self.fileIndex >= self.fileCount:
                # there is no more new files
                raise StopIteration
            
            # process previous file
            currentFileCursor = self.sampleCountPerFile - self.remainDataInThisFile
            dataInPreviousWindow = self.data[currentFileCursor:,:]
            remainWindowSize = self.windowSize - self.remainDataInThisFile
            baseTime1 = pathToTime(self.filepath)
            timeDelta1 = timedelta(seconds=(currentFileCursor / self.sr))

            # start the new file but all variables are local
            filepath = self.filepaths[self.fileIndex]
            # sr, data = wavfile.read(filepath)
            data, sr = librosa.load(self.filepath, sr=48000, mono=False)
            data = data.T
            sampleCountPerFile = data.shape[0]

            if self.sr != sr:
                raise ValueError(f"the sampling rate of file {self.filepath} is {sr} not equal to {self.sr}")
                        
            if self.sampleCountPerFile != sampleCountPerFile:
                raise ValueError(f"the sample count per file of file {self.filepath} is {sampleCountPerFile} not equal to {self.sampleCountPerFile}")
                
            # extract the data
            remainDataInThisFile = sampleCountPerFile
            currentFileCursor = sampleCountPerFile - remainDataInThisFile
            dataInCurrentWindow = data[currentFileCursor:currentFileCursor+remainWindowSize,:]
            if dataInPreviousWindow.shape[0] == 0:
                currentWindow = dataInCurrentWindow
            else:
                # print(dataInPreviousWindow.shape)
                # print(dataInCurrentWindow.shape)
                currentWindow = np.concatenate((dataInPreviousWindow, dataInCurrentWindow), axis=0)
            # update variables
            self.remainDataInThisFile -= self.stepSize
            
            baseTime2 = pathToTime(filepath)
            timeDelta2 = timedelta(seconds=((currentFileCursor+remainWindowSize) / self.sr))
            self.worldTime = [baseTime1+timeDelta1, baseTime2+timeDelta2]
            
        else:
            # remaining size of the current file is larger than the window size. We do not need to access other files
            currentFileCursor = self.sampleCountPerFile - self.remainDataInThisFile

            baseTime = pathToTime(self.filepath)
            timeDelta1 = timedelta(seconds=(currentFileCursor / self.sr))
            timeDelta2 = timedelta(seconds=((currentFileCursor + self.windowSize) / self.sr))
            self.worldTime = [baseTime+timeDelta1, baseTime+timeDelta2]
            
            currentWindow = self.data[currentFileCursor:currentFileCursor+self.windowSize, :]
            self.remainDataInThisFile -= self.stepSize
        self.sampleIndex += self.windowSize
        self.windowIndex += 1
        
        self.time += self.timeDelta
        # sampleIndex is the start sample index of the window (e.g. 0, 48000, 96000.....)
        # windowIndex is the window index (e.g. 0, 1, 2.....)
        # time is a multiple of windowIndex with stepDuration (e.g. 0 Seconds, 0.5 Seconds.....)
        # worldTime is the start and end real-world time of the window
        return currentWindow, self.sampleIndex - self.windowSize, self.windowIndex - 1, self.time, self.worldTime
    
    def pathToTime(self, filepath):
        filename = pathToName(filepath)
        timeToken = '_'.join(filename.split('.')[0].split('_')[2:])
    #     dry_run_04_16_2022_00_00_02_054049.wav
        timeDatetime = datetime.strptime(timeToken, "%m_%d_%Y_%H_%M_%S_%f")
        return timeDatetime

def checkDataAroundTimeWindow(timeWindowWorldTime, filepaths):
    """
    plot data stream, spectrogram and load audio around a time window
    
    Args:
        timeWindowWorldTime: a list consists of the start and end worldtime of a timewindow
        filepaths: filepath list which consists of the time window
        
    Returns:
        None
    """
    res = []
    timeStart = timeWindowWorldTime[0]
    timeEnd = timeWindowWorldTime[1]
    for i,filepath in enumerate(filepaths):
        if timeStart < pathToAccurateTime(filepath) and len(res) == 0:
            res.append(filepaths[i-1])
        if timeEnd < pathToAccurateTime(filepath):
            if filepaths[i-1] not in res:
                res.append(filepaths[i-1])
            else:
                pass
            break
    if i >= 0:
        visualizeFiles(res, play=True)
    else:
        return

def checkDuplicates(filepaths):
    """
    Check duplicates in the provided list of filepaths

    Returns:
        dup: the duplication list
        isDup: a flag indicates if there are duplicates in the provided list
    """
    # ref: https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    seen = set()
    dup = []
    for x in filepaths:
        if x in seen:
            dup.append(x)
        else:
            seen.add(x)
    return dup, len(dup) != 0

def getDatasetStat(filepaths):
    startDate = filepaths[0]
    endDate = filepaths[-1]
    fileCount = len(filepaths)
    filepath = filepaths[0]
    sr, x = wavfile.read(filepath)
    sampleCount = x.shape[0]
    duration = sampleCount / sr
    datasetDuration = fileCount * duration
    return DatasetStat(startDate, endDate, fileCount, datasetDuration)
    
def getFileStat(filepath):
    sr, x = wavfile.read(filepath)
    # sampleRate
    sr = sr
    # sample count
    sampleCount = x.shape[0]
    # channel count
    channelCount = x.shape[1]
    # duration in second
    duration = sampleCount / sr
    return FileStat(filepath = filepath, sr=sr, sampleCount = sampleCount, channelCount = channelCount, duration = duration)

# Filename related helper functions start
def getAllFilePaths(data_dirs):
    """
    Return all filenames in the data_dirs
    """
    filenames = []
    for data_dir in data_dirs:
        filenames += sorted([join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))])
    return filenames

def getAllFilePathByDay(filenames_cleaned):
    """
    seperate files based on the day they are recorded
    return: 
        list of list
    you need to map the day by yourself
    """
    filenames_by_day = []
    temp = []
    current = -1 
    prev = -1
    # we can also use a dict
    for filename in filenames_cleaned:
        current = int(filename.split('_')[3]) # get date
        if current != prev and prev != -1:
            filenames_by_day.append(temp)
            temp = []
        temp.append(filename)
        prev = current
    filenames_by_day.append(temp)
    return filenames_by_day

def printFilenames(filenames):
    for filename in filenames:
        print(filename)

# print out filenames in each day with their data format
def printFilePathByDay(filenames_by_day):
    i = 0
    for filename_day in filenames_by_day:
        samplerate, data = wavfile.read(filename_day[0])
        print("filename:",filename_day[0], "data_shape:",np.shape(data), "size", len(filename_day), "index", i)
        i = i + len(filename_day)
        
# Save and load helper functions start

def loadFile(filename):
    res = []
    with open(filename, "rb") as fp: 
        res = pickle.load(fp)
    return res

def saveFile(filename, data):
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + filename
    with open(filename, "wb") as fp: 
        res = pickle.dump(data, fp)
    return

def removeFiles(removeList):
    for filename in removeList:
        process = subprocess.Popen(["rm", filename])
        sts = os.waitpid(process.pid, 0)
# Save and load helper functions end

def copyFiles(destination, filenames):
    for filename in filenames:
        process = subprocess.Popen(["cp", filename, destination])
        sts = os.waitpid(process.pid, 0)

def saveCSV(filename, data):
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + filename + ".csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


        
        
        
# For dataset cleaning

def countSampleLargerThanThreshold(filename, ch, th):
    sr, x = wavfile.read(filename)
    x = x.astype(np.float64)
    x = np.transpose(x)
    channelData = x[ch, :]
    res = (channelData > th).sum() / len(channelData)
    print(res)
    return res

def countSampleLargerThanThresholdWithChannel(channelData, th):
    """
    check the percentage of sample points larger than the provided threshold in a provided channelData numpy array
    Reason:
    1. we do not use max because there might be noise (like pulse)
    2. we do not use average because the sum is always close to zero (data is AC)
    3. RMS may be another choice
    """
    res = (channelData > th).sum() / len(channelData)
    return res

# countSampleLargerThanThreshold('/data3/beitong2/dataset/dry_run_04_19_2022_12_11_29_074658.wav', 2, 5000)
# countSampleLargerThanThreshold('/data3/beitong2/dataset/dry_run_03_28_2022_15_11_57_432842.wav', 2, 5000)
# countSampleLargerThanThreshold('/data3/beitong2/dataset/dry_run_04_19_2022_12_05_38_822760.wav', 0, 8000)
# countSampleLargerThanThreshold('/data3/beitong2/dataset/dry_run_04_23_2022_02_32_08_702806.wav', 0, 8000)

# Data correction helper functions start
def correctData(filenames, output_path, sr=48000):
    """
    correctData with filename list
    Workflow:
        1. run this script
        2. all corrected files will be stored in the output_path
        3. all files which should be removed are stored in the returned removeList
        4. visualize correctData (see if they are correct now)
        5. visualize files in the removeList (see if we really want to remove them)
        6. mv corrected files to the original dataset (overwrite)
        7. rm files in the removeList
        8. work on data in another day
    """
    removeList = []
    counter = 0
    totalCount = len(filenames)
    
    for filename in filenames:
        counter +=1 
        main_token = filename.split('/')[-1]
        sr, x = wavfile.read(filename)
        out = x
        x = x.astype(np.float64)
        ch1 = x[:,0]
        ch2 = x[:,1]
        ch3 = x[:,2]
        ch4 = x[:,3]

        # test fft
        x = np.transpose(x)
        N = 4096
        
#         # plot fft res
#         plt.plot(fftfreq(N, 1/48000)[:N//2], np.abs(fft(ch1, n=N)[0:N//2]))
#         plt.show()
#         plt.plot(fftfreq(N, 1/48000)[:N//2], np.abs(fft(ch2, n=N)[0:N//2]))
#         plt.show()
#         plt.plot(fftfreq(N, 1/48000)[:N//2], np.abs(fft(ch3, n=N)[0:N//2]))
#         plt.show()
#         plt.plot(fftfreq(N, 1/48000)[:N//2], np.abs(fft(ch4, n=N)[0:N//2]))
#         plt.show()
#         print(fftfreq(N, 1/48000)[:N//2][:10])

        ch1FFTRes = np.abs(fft(ch1, n=N)[0:N//2])
        ch1FFTResCheck = ch1FFTRes[5].sum()/ch1FFTRes.sum()
        ch2FFTRes = np.abs(fft(ch2, n=N)[0:N//2])
        ch2FFTResCheck = ch2FFTRes[5].sum()/ch2FFTRes.sum()
        ch3FFTRes = np.abs(fft(ch3, n=N)[0:N//2])
        ch3FFTResCheck = ch3FFTRes[5].sum()/ch3FFTRes.sum()
        ch4FFTRes = np.abs(fft(ch4, n=N)[0:N//2])
        ch4FFTResCheck = ch4FFTRes[5].sum()/ch4FFTRes.sum()
        allFFTRes = [ch1FFTResCheck,ch2FFTResCheck,ch3FFTResCheck,ch4FFTResCheck]
        index = allFFTRes.index(max(allFFTRes))
#         print(ch1FFTResCheck,ch2FFTResCheck,ch3FFTResCheck,ch4FFTResCheck)

        length = 48000
        ch1PercentF = countSampleLargerThanThresholdWithChannel(ch1[:length], 6000) > 0.001
        ch1PercentE = countSampleLargerThanThresholdWithChannel(ch1[-length:], 6000) > 0.001
        ch2PercentF = countSampleLargerThanThresholdWithChannel(ch2[:length], 6000) > 0.001
        ch2PercentE = countSampleLargerThanThresholdWithChannel(ch2[-length:], 6000) > 0.001
        ch3PercentF = countSampleLargerThanThresholdWithChannel(ch3[:length], 6000) > 0.001
        ch3PercentE = countSampleLargerThanThresholdWithChannel(ch3[-length:], 6000) > 0.001
        reason = 0

        if ch1PercentF != ch1PercentE or  ch2PercentF != ch2PercentE or ch3PercentF != ch3PercentE:
            # halfly recorded
            reason = 9
            removeList.append(filename)
            continue

        if index == 0:
            reason = 1
            order = [2,3,0,1]
            x = x[order]
        elif index == 1:
            reason = 2
            order = [3,0,1,2]
            x = x[order] 
        elif index == 3:
            reason = 3
            order = [1,2,3,0]
            x = x[order] 

        if reason == 0:
            print(f"{filename} should not be in this filelist")
            continue

        print(f"{filename} reason: {reason} {counter} out of {totalCount}")

        x = x.astype(np.int16)
        wavfile.write(f"{output_path}/{main_token}", sr, np.transpose(x))
        
# Data correction helper functions end

