from collections import defaultdict
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from pydub import AudioSegment
from scipy.io import wavfile
import spectrogram 
import time
import torch
import torchvision


AudioSegment.converter = "ffmpeg"

def readAudio(mp3_path):
    # files             
    dst = mp3_path[:-4] + ".wav"

    # convert mp3 to wav   
    if not os.path.exists(dst):                                                         
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(dst, format="wav")

    # read wav and convert to spectrogram
    data,sr = wavfile.read(dst)
    
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 15000  # Hz # High cut for our butter bandpass filter
    
    data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
    
    wav_spectrogram = pretty_spectrogram(
        data.astype("float64"),
        fft_size = 2048,
        step_size = fft_size//16,
        log = True,
        thresh = 4,
    )

    return data, sr

class AccentDataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_path):
        self.data_names = []
        self.classes = []
        self.encod_data = []

        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            for row in csv_reader:
                if (row[8] == "FALSE"):
                    mp3_path = os.path.join(path,row[3]+'.mp3')
                    self.data_names.append((mp3_path, row[4]))
                    if (row[4] not in self.classes):
                        self.classes.append(row[4])

            for data in self.data_names:
                label = torch.zeros(len(self.classes)) 
                label[self.classes.index(data[1])] = 1
                self.encod_data.append((data[0],label))

    def __getitem__(self,index):
        dat, s_rate = readAudio(self.data_names[index][0])
        label = self.encod_data[index][1]
        return dat, label

    def __len__(self):
        return len(all_data)
    