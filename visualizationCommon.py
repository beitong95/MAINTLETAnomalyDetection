import librosa
from IPython.display import Audio # needed for correct audio player
from IPython.core.display import display
from MaintletTimer import *
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
# this file is specially for the dataset we collected in the mechanical room
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-poster')
import time 
import gc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def saveFigure(name):
    plt.savefig(name)

def saveFigureAsPDF(name):
    plt.savefig(name + ".pdf", format = "pdf")

def loss_plot(loss, val_loss):
    """
    Plot loss curve.
    loss : list [ float ]
        training loss time series.
    val_loss : list [ float ]
        validation loss time series.
    return   : None
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title("Model loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train", "Validation"], loc="upper right")

def rocPlot(yTrue, yPred):
    # two class
    fpr, tpr, _ = roc_curve(yTrue, yPred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
 
def confusionMatrixPlot(yTrue, decision, labels=[0,1]):
    cm = confusion_matrix(yTrue, decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot()
    plt.show()
 
def plotMelSpectrogram(S, sr=48000, hop_length=512):
    """
    Plot mel spectrogram given mel spectrogram result
    Useful for checking original mel spectrogram and generated mel spectrogram
    """
    # S shape: nFrame, nMel
    vectors = S
    fig, ax = plt.subplots(figsize=(5,5))
    librosa.display.specshow(vectors.T, y_axis='linear', sr=sr, hop_length=hop_length,x_axis='time', ax=ax)
    plt.show()
    
# !!! this is a depracated function 
# use visualizeFiles instead
def visualizeFile(filepath, sr=48000, play=False): 
    """
    Plot an audio file give the filename (actually the filename is a path)
    Load the player to play this audio file is play is set to True
    """
    x, sr = librosa.load(filepath, sr=sr, mono=False)
    print(f"shape of data {x.shape}")
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(36,12))
    ch1 = x[0,:]
    librosa.display.waveshow(ch1, sr=sr, ax=ax[0])
    ch2 = x[1,:]
    librosa.display.waveshow(ch2, sr=sr, ax=ax[1])
    ch3 = x[2,:]
    librosa.display.waveshow(ch3, sr=sr, ax=ax[2])
    ch4 = x[3,:]
    librosa.display.waveshow(ch4, sr=sr, ax=ax[3])
    if play:
        display(Audio(ch1, rate=sr))
        display(Audio(ch2, rate=sr))
        display(Audio(ch3, rate=sr))
        display(Audio(ch4, rate=sr))

# !!! this is a depracated function
# we can use '%matplotlib widget' to zoom in interactivaly 
def visualizePartOfFile(filename, start=0, end=10, sr=48000, play=False): 
    """
    Plot part of a audio file (this is a more general version of plotFileWithLibrosa)
    Arguments:
        - filename: a string represents the path of the audio file
        - start: the start timestamp in seconds (you can use float number)
        - end: the end timestamp in seconds
        - sr: sampleing rate
        - play: if play the file
    """

    sr, x = wavfile.read(filename)
    x = x.astype(np.float64)
    x = np.transpose(x)
    sampleCount = x.shape[1]
    
    start = max(0, start * 48000)
    end = min(sampleCount - 1, end * 48000 - 1)
    start = int(start)
    end = int(end)
    
    if start < 0 or end > x.shape[1]:
        return
    print(f"shape of data {x.shape}")
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(36,12))
    ch1 = x[0,start:end]
    librosa.display.waveshow(ch1, sr=sr, ax=ax[0])
    ch2 = x[1,start:end]
    librosa.display.waveshow(ch2, sr=sr, ax=ax[1])
    ch3 = x[2,start:end]
    librosa.display.waveshow(ch3, sr=sr, ax=ax[2])
    ch4 = x[3,start:end]
    librosa.display.waveshow(ch4, sr=sr, ax=ax[3])
    plt.show()
    
    if play:
        display(Audio(ch1, rate=sr))
        display(Audio(ch2, rate=sr))
        display(Audio(ch3, rate=sr))
        display(Audio(ch4, rate=sr))

def visualizeWindow(window, sr=48000):
    """
    Visualize window with raw plots, spectrograms and audios
    
    Args:
        window: the time window of a single sensor stream
    """
    
    channelData = window
    
    # plot original data
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,3))
    axs[0].set_ylabel('amplitude')

    librosa.display.waveshow(channelData, sr=sr, ax=axs[0])
    
    S = librosa.feature.melspectrogram(y=channelData, sr=sr, n_mels=128, fmax=24000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=24000, ax=axs[1])
    
    plt.tight_layout()
    plt.show()
    
    display(Audio(channelData, rate=sr))   
        
def visualizeFiles(filepaths, downSampleStep = 1, play=False, verbose=True): 
    """
    Visualize files with raw plots, spectrograms and audios
    """
    # parameters
    if len(filepaths) > 500:
        print("Parameter Error: file count in filenames should be smaller than 500")
    
    if len(filepaths) > 20:
        play = False
    startTime = time.time()
    downSampleStep = min(30, max(1, downSampleStep))
    channelCount = 0
    if verbose:
        print(f"start: {filepaths[0]}")
        print(f"end  : {filepaths[-1]}")
    # load and concatenate
    with MaintletTimer("prepare data") as mt:
        fileCount = len(filepaths)
        if fileCount <= 0:
            return
        filepath = filepaths[0]
        sr, data = wavfile.read(filepath)
        sampleCountReference = data.shape[0]
        
        totalSampleCount = sampleCountReference * fileCount
        channelCountReference = data.shape[1]
        channelCount = channelCountReference
        dataBuffer = np.empty((totalSampleCount, channelCountReference))
        counter = 0
        for filepath in filepaths:
            _, data = wavfile.read(filepath)
            sampleCountInThisFile = data.shape[0]
            channelCountInThisFile = data.shape[1]
            if sampleCountInThisFile != sampleCountReference or channelCountInThisFile != channelCountReference:
                # check 1, sample count and channel count
                print(f"{filename}: Incorrect sample or channel count")
                continue
            dataBuffer[counter*sampleCountReference:(counter+1)*sampleCountReference, :] = data
            counter += 1

    # downsample
    if play == True:
        downSampleStep = 1
    dataBufferOriginal = np.array(dataBuffer) # keep the 
    dataBuffer = dataBuffer[::downSampleStep,:]
    srForPlot = int(sr/downSampleStep)
    
    # plot original data
    with MaintletTimer("plot data") as mt:
        x = np.transpose(dataBuffer)
        xOriginal = np.transpose(dataBufferOriginal)
        # print(f"shape of data {x.shape}")
        fig, axs = plt.subplots(nrows=channelCount, ncols=2, figsize=(18,8))
        for i in range(channelCount):
            # fig.suptitle('Original Data', fontsize=32)
            channelData = x[i,:]
            channelDataOriginal = xOriginal[i,:]
            librosa.display.waveshow(channelData, sr=srForPlot, ax=axs[i][0])
            S = librosa.feature.melspectrogram(y=channelDataOriginal, sr=sr, n_mels=128, fmax=24000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            print(S.shape)
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=24000, ax=axs[i][1])
            axs[i][0].set_ylabel('amplitude')
            if i != channelCount-1:
                axs[i][0].set_xlabel('')
                axs[i][1].set_xlabel('')
            
        plt.tight_layout()
        plt.show()
    
    if play:
        with MaintletTimer("prepare audio") as mt:
            for i in range(channelCount):
                channelData = xOriginal[i,:]
                display(Audio(channelData, rate=sr))
        
    elapsedTime = time.time() - startTime

    return elapsedTime

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Reference: stackoverflow"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def visualizeDataset(filenames, chunkSize, downSampleStep=10):
    """
    Plot more files for the entire dataset checking with progress tracking
    """
    fileCount = len(filenames)
    chunkCount = int(fileCount / chunkSize)
    counter = 0
    avgElapsedTime = 0
    filenameChunks = list(chunks(filenames, chunkSize))
    for filenameChunk in filenameChunks:
        elapsedTime = visualizeFiles(filenameChunk, downSampleStep=downSampleStep,verbose=False)
        counter += 1
        
        # calculate avg elapsed time
        if avgElapsedTime == 0:
            avgElapsedTime = elapsedTime
        else:
            avgElapsedTime = (avgElapsedTime * (counter-1) + elapsedTime) / counter
            
        finishedFileCount = counter * chunkSize
        print(f"Start: {filenameChunk[0]}")
        print(f"End  : {filenameChunk[-1]}")

        print(f"Data Index [{finishedFileCount - chunkSize},{finishedFileCount-1}]")
        print(f"Progress {finishedFileCount} out of {fileCount}. Use {round(elapsedTime, 2)} S. Avg Process Time {round(avgElapsedTime, 2)} S. Remaining Time {round(avgElapsedTime * (chunkCount - counter),2)} S")
        gc.collect()