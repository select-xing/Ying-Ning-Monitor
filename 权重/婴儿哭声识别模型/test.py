import pyaudio
import wave
from data_process import AudioConfig
import numpy as np
import os
import torch
import torchaudio
import sys
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchaudio.transforms import MelSpectrogram, Resample, TimeStretch, FrequencyMasking, TimeMasking
import matplotlib.pyplot as plt
import torch.nn as nn
from Resnet_DualAttention import ResNet50DualAttention
config = AudioConfig()
# 参数设置
FORMAT = pyaudio.paInt16  # 16位采样深度
CHANNELS = 1              # 单声道
RATE = 44100              # 采样率（Hz）
RECORD_SECONDS = 5        # 录制时长
CHUNK = 1024              # 数据块大小

p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("开始录制...")
frames = []

# 读取并保存3秒音频数据
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录制完成")

# 关闭流并保存为WAV文件
stream.stop_stream()
stream.close()
p.terminate()

# 写入文件
with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

filepath = "D:/pycharmproject/baby_cry_classify/output.wav"
# 读取音频并重采样
waveform, orig_sr = torchaudio.load(filepath)
if orig_sr != config.sample_rate:
    resampler = Resample(orig_sr, config.sample_rate)
    waveform = resampler(waveform)

# 统一音频长度（裁剪/填充）
target_length = int(config.duration * config.sample_rate)
if waveform.shape[1] > target_length:
    start = np.random.randint(0, waveform.shape[1] - target_length)
    waveform = waveform[:, start:start + target_length]
else:
    padding = target_length - waveform.shape[1]
    waveform = torch.nn.functional.pad(waveform, (0, padding))
waveform = waveform.mean(dim=0, keepdim=True)

# 转换为梅尔频谱图
mel_spec = MelSpectrogram(
    sample_rate=config.sample_rate,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    n_mels=config.n_mels,
    power=config.power
)(waveform)


# 对数压缩
log_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-8)  # Min-Max归一化
log_spec = log_spec.repeat(3, 1, 1)
log_spec = log_spec.unsqueeze(0)
#model = AudioClassifier()
#model.resnet.fc = nn.Linear(2048, 6)
model = ResNet50DualAttention()
model.fc = nn.Linear(2048, 6)
# final_model.resnet.fc = nn.Linear(2048, 6).to(device)
# final_model = models.resnet18(weights=None).to(device)
# 步骤2：加载参数
# final_model = models.resnet18(weights=None).to(device)

# 步骤2：加载参数
state_dict = torch.load("best_BiLSTM_model2_val.pth")
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

outputs = model(log_spec)
_, predicted = torch.max(outputs, 1)
categories = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
print(f"婴儿哭的原因是{categories[int(predicted)]}")

