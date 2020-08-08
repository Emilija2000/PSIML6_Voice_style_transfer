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
import soundfile as sf


AudioSegment.converter = "ffmpeg"
#allowed_genres = ["blues","classical","country","disco","hiphop","metal","pop","reggae","rock"]
allowed_genres = ["rock","classical","reggae"]

def readAudio(mp3_path):
    '''
    Read audio file on location mp3_path and creates spectrogram
    '''
    y, sr = librosa.load(mp3_path)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=220)
    melSpec_dB = librosa.power_to_db(melSpec)
    #librosa.display.specshow(melSpec_dB)

    return melSpec_dB, sr

def writeAudio(spectrogram, sample_rate, mean, var,output_file_name):
    '''
    For given normalized mel spectrogram, sample_rate, mean and variance of
    the original recording, write .wav audiofile with the name output_file_name + '.wav'
    '''
    spectrogram=spectrogram*var+mean
    spectrogram = librosa.db_to_power(spectrogram)    
    
    # write output
    file_name = output_file_name + '.wav'

    audio_signal = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate)
    sf.write(file_name, audio_signal, sample_rate)
    

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_path):
        self.data_names = []    #mp3 file location and data label
        self.data_names_onehot = [] #mp3 file location and onehot encoded data label
        self.classes = [] #all classes (possible labels)
        
        self.data = [] #tuples - ((spectrogram,sample_rate,mean,var), label) - labels are not one hot
        
        with open(csv_path) as csvfile:
            #start cvs reader for our dataset
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
                #calculate mean and var
                mean_spect = spect.mean()
                var_spect = spect.var()
                spect = (spect-mean_spect)/var_spect
                #save data
                self.data.append(((spect,sr,mean_spect,var_spect),label))
                i +=1
                #print
                if i%50 == 0:
                    print('Loading: ',i, ' items from dataset')

    def __getitem__(self,index):
        filename = self.data_names[index][0]
        spect = self.data[index][0][0]
        mean = self.data[index][0][2]
        var = self.data[index][0][3]
        label = self.data[index][1]
        return filename,spect, mean, var, label

    def __len__(self):
        return len(self.data_names)
       