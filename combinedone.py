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
    
def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)
    
model = init_onnx_model(model_path='./model.onnx')


def validate(model,inputs: torch.Tensor):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs[0]

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze() 
    return sound

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250

import threading


continue_recording = True

def start_recording1(threadname):
    import config
    stream1 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=1)
    data1 = []
    voiced_confidences1 = []
    global new_confidence1
    continue_recording1 = True
    #pp1 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")
    
    while continue_recording1:
        audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
        data1.append(audio_chunk1)
        audio_int161 = np.frombuffer(audio_chunk1, np.int16)
        audio_float321 = int2float(audio_int161)
        vad_outs1 = validate(model, torch.from_numpy(audio_float321))
        new_confidence1 = vad_outs1[:,1].numpy()[0].item()
        config.nc1=new_confidence1
        voiced_confidences1.append(new_confidence1)
        #pp1.update(new_confidence1)
    #pp1.finalize()
    
    
import threading


def start_recording2(threadname):
    import config
    stream2 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=2)
    data2 = []
    voiced_confidences2 = []
    global new_confidence2
    continue_recording2 = True
    #pp2 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")
    
    while continue_recording2:
        audio_chunk2 = stream2.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
        data2.append(audio_chunk2)
        audio_int162 = np.frombuffer(audio_chunk2, np.int16)
        audio_float322 = int2float(audio_int162)
        vad_outs2 = validate(model, torch.from_numpy(audio_float322))
        new_confidence2 = vad_outs2[:,1].numpy()[0].item()
        config.nc2=new_confidence2
        voiced_confidences2.append(new_confidence2)
        #pp2.update(new_confidence2)
    #pp2.finalize()

import io
import socket
import struct
from PIL import Image
import cv2
import numpy
import sys
import time

def decisionblockforcameraone(threadname):
    while True:
        import config
        if(config.nc1>0.72):
            if(config.db1>-28):
                config.cameraoneon=True
                while(config.nc1>0.75):
                    config.cameraoneon=True
                    time.sleep(2)
                    if(config.nc1<0.75):
                        time.sleep(5)
                time.sleep(5)
        else:
            config.cameraoneon=False
def decisionblockforcameratwo(threadname):
    while True:
        import config
        if(config.nc2>0.72):
            if(config.db2>-28):
                config.cameratwoon=True
                while(config.nc2>0.75):
                    config.cameratwoon=True
                    time.sleep(2)
                    if(config.nc2<0.75):
                        time.sleep(5)
                time.sleep(5)
        else:
            config.cameratwoon=False    
        
def start_webcam(threadname):

    i=0
    
    img1 = None
    img2 = None
    global db1
    global db2
    '''
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 5454))  
    server_socket.listen(0)
    print("Listening")
    connection = server_socket.accept()[0].makefile('rb')
    vid = cv2.VideoCapture(0
    '''
    vid1=cv2.VideoCapture(1)
    vid2=cv2.VideoCapture(2)
    while True:
        #image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

        '''
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        im1 = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        ret, im2 = vid.read()


        image_stream2 = io.BytesIO()
        image_stream2.write(connection.read(image_len))
        image_stream2.seek(0)
        image2 = Image.open(image_stream2)
        im2 = cv2.cvtColor(numpy.array(image2), cv2.COLOR_RGB2BGR)
        

        from combineoftwo import db1
        from combineoftwo import db2
        print("db1:",db1)
        print("db2:",db2)
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break

        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        im1 = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        ret, im2 = vid.read()
        '''

        rval1, im1 = vid1.read()
        rval2, im2 = vid2.read()
        import config
        #print("db1:",config.db1)
        #print("db1:",config.db2)
        #print("nc1:",config.nc1)
        #print("nc2:",config.nc2)
        if(config.cameraoneon==True and config.cameratwoon==True):
            #im1=cv2.imread('./speakerone.jpg')
            #im2=cv2.imread('./speakertwo.jpg')
            images_1_2_h = np.hstack((im1, im2))
            cv2.imshow('Video',images_1_2_h)
            #print("thisconditionworks")
            i=0
        if(config.cameraoneon==True and config.cameratwoon==False):
            #im1=cv2.imread('./speakerone.jpg')
            cv2.imshow('Video',im1)
            #print("thisconditionworkstwo")
            i=1
        if(config.cameratwoon==True and config.cameraoneon==False):
            #im2=cv2.imread('./speakertwo.jpg')
            #print("thisconditionworksthree")
            cv2.imshow('Video',im2)
            i=2
        if(config.cameraoneon==False and config.cameratwoon==False):
            im3=cv2.imread('./noone.jpg')
            #print("noworks")
            cv2.imshow('Video',im3)
            i=3

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




