import time
from os import path
from pydub import AudioSegment
import copy
from collections import defaultdict
import numpy as np
import torch
import torchaudio
import torchvision
import matplotlib.pyplot as plt
import csv
import os

AAudioSegment.converter = "ffmpeg"
torchaudio.set_audio_backend = "SoundFile"

def readAudio(mp3_path):
    # files             
    dst = mp3_path[:-4] + ".wav"

    # convert mp3 to wav   
    if not os.path.exists(dst):                                                         
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(dst, format="wav")

    test,sr = torchaudio.load_wav(dst)

    return test, sr

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
    