import time
from os import path
from pydub import AudioSegment
import copy
from collections import defaultdict
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

AudioSegment.converter = "C:\\ffmpeg-20200802-b48397e-win64-static\\ffmpeg-20200802-b48397e-win64-static\\bin\\ffmpeg"

torchaudio.set_audio_backend = "SoundFile"

if __name__=='__main__':
    # files                                                                         
    src = "D:\\Accents\\4114_6391_bundle_archive\\recordings\\recordings\\albanian5.mp3"
    dst = "test.wav"

# convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    test,sr = torchaudio.load_wav(dst)
    print(test)
    