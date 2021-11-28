import io
import numpy as np
import torch
torch.set_num_threads(1)
#import torchaudio
import matplotlib
import matplotlib.pylab as plt
#torchaudio.set_audio_backend("soundfile")
import pyaudio
global new_confidence1
global new_confidence2
new_confidence1=0
new_confidence2=0
import config
import onnxruntime

import io
import socket
import struct
from PIL import Image
import cv2
import numpy
import sys
import time
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250



continue_recording = True
