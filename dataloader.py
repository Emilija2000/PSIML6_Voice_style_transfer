from collections import defaultdict
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from pydub import AudioSegment
import scipy
import scipy.signal
from scipy.io import wavfile
import time
import torch
import torchvision
import librosa                    
import librosa.display
from PIL import Image
from matplotlib import cm



AudioSegment.converter = "ffmpeg"
allowed_genres = ["blues","classical","country","disco","hiphop","metal","pop","reggae","rock"]

def readAudio(mp3_path):
    '''
    Read audio file on location mp3_path using librosa and creates spectrogram
    '''
    y, sr = librosa.load(mp3_path)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    #librosa.display.specshow(melSpec_dB)
    return melSpec_dB, sr

def writeAudio(spectrogram, sample_rate, output_file_name):
    '''
    For given scipy spectrogram and sample_rate, write .wav audiofile
    with the name output_file_name + '.wav'
    '''
    audio_signal = librosa.core.spectrum.griffinlim(spectrogram)
    #print(audio_signal, audio_signal.shape)
    # write output
    file_name = output_file_name + '.wav'
    scipy.io.wavfile.write(file_name, sample_rate, np.array(audio_signal, dtype=np.int16))

def readSpectrogram(png_path):
    '''
    Reads png file and save spectrogram as a matrix
    '''
    pass

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_path):
        self.data_names = []    #mp3 file location and data label
        self.data_names_onehot = [] #mp3 file location and onehot encoded data label
        self.classes = [] #all classes (possible labels)
        
        self.data = [] #tuples - ((spectrogram,sample_rate), label) - labels are not one hot
        
        with open(csv_path) as csvfile:
            #start cvs reader
            csv_reader = csv.reader(csvfile,delimiter=',')
            next(csv_reader) #skip first row
            for row in csv_reader:
                if (row[59] in allowed_genres): 
                    mp3_path = os.path.join(path,row[59],row[0])
                    self.data_names.append((mp3_path, row[59])) 
                    if (row[59] not in self.classes):
                        self.classes.append(row[59])

            #create onehot encoding for data labels
            i = 0
            for data in self.data_names:
                #saving just datafile names and labels as onehot
                label_onehot = torch.zeros(len(self.classes)) 
                label_onehot[self.classes.index(data[1])] = 1
                self.data_names_onehot.append((data[0],label_onehot))
                
                #int encoding for classes
                label = torch.argmax(label_onehot).item()
                
                #reading data and saving it as spectrograms and labels as int!
                spect, sr = readAudio(data[0])
                #reshape spectrogram to represent 1 channel
                spect = spect[:,:1280]     
                #spect = Image.fromarray(np.uint8(spect))        
                #spect = torchvision.transforms.Grayscale(3)(spect)
                spect = torch.Tensor(spect).reshape(1,spect.shape[0],spect.shape[1])
                #save data
                self.data.append(((spect,sr),label))
                i +=1
                #print
                if i%50 == 0:
                    print('Loading: ',i, ' items from dataset')

    def __getitem__(self,index):
        return self.data_names[index][0],self.data[index][0][0], self.data[index][1]

    def __len__(self):
        return len(self.data_names)
       