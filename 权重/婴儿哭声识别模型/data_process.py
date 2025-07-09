import os
import torch
import torchaudio
import sys
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchaudio.transforms import MelSpectrogram, Resample, TimeStretch, FrequencyMasking, TimeMasking
import matplotlib.pyplot as plt


class AudioConfig:
    # 基础配置
    sample_rate = 16000          # 目标采样率[1](@ref)
    duration = 5.0               # 统一音频时长（秒）
    n_fft = 512                  # FFT窗口大小（STFT参数）[4](@ref)
    hop_length = 256             # 帧移（与TimeStretch共用参数）[1](@ref)
    n_freq = n_fft // 2 + 1               # STFT滤波器组数量（新增关键参数）[1](@ref)
    n_mels = 64                  # 梅尔带数（仅用于MelSpectrogram）[6](@ref)
    power = 2.0                  # 频谱图幂次

    # 数据增强参数
    time_stretch = 0.2           # 时间拉伸范围（±20%）[1](@ref)
    freq_mask = 15                # 频率掩蔽带宽（梅尔带数单位）[6](@ref)
    time_mask = 70                # 时间掩蔽长度（时间帧单位）[6](@ref)


def visualize_spectrogram(log_spec, waveform, sample_rate, hop_length):
    """可视化梅尔频谱图与原始波形对比"""
    # 数据格式转换
    spec_np = log_spec[0].numpy()  # 取首个通道 (3通道重复数据)
    time_axis = np.arange(spec_np.shape[1]) * hop_length / sample_rate
    freq_axis = np.linspace(0, sample_rate / 2, spec_np.shape[0])
    # 创建配置对象
    config = AudioConfig()

    # 创建画布
    plt.figure(figsize=(15, 8))

    # 子图1：原始波形
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, config.duration, waveform.shape[1]), waveform[0].numpy())
    plt.title("Waveform Visualization")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, config.duration)

    # 子图2：梅尔频谱图
    plt.subplot(2, 1, 2)
    img = plt.imshow(spec_np,
                     aspect='auto',
                     origin='lower',
                     extent=[time_axis[0], time_axis[-1],
                             freq_axis[0], freq_axis[-1]],
                     cmap='magma')
    plt.colorbar(img, label='Normalized Energy (dB)')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

class Baby_Cry_Dataset(Dataset):
    def __init__(self, file_list, labels, config, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.augment = augment

        # 初始化数据增强变换
        self.time_stretch = TimeStretch(config.hop_length, config.n_mels)
        self.freq_mask = FrequencyMasking(config.freq_mask)
        self.time_mask = TimeMasking(config.time_mask)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 预处理音频
        spec = audio_preprocessing(self.file_list[idx], self.config)
        # 数据增强
        if self.augment:
            # 频率掩蔽
            if np.random.rand() > 0.5:
                spec = self.freq_mask(spec)

            # 时间掩蔽
            if np.random.rand() > 0.5:
                spec  = self.time_mask(spec)
        return spec, self.labels[idx]

def audio_preprocessing(filepath, config):
    """音频预处理流水线"""
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
    visualize_spectrogram(log_spec, waveform,config.sample_rate, config.hop_length)
    return log_spec


def prepare_datasets(data_root):
    # 获取所有文件路径和标签
    categories = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
    #categories = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10',]
    file_paths = []
    labels = []

    for idx, category in enumerate(categories):
        category_dir = os.path.join(data_root, category)
        for file in os.listdir(category_dir):
            if file.endswith('.wav'):
                file_paths.append(os.path.join(category_dir, file))
                labels.append(idx)

    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels)

    # 创建配置对象
    config = AudioConfig()

    # 创建数据集实例
    train_dataset = Baby_Cry_Dataset(X_train, y_train, config, augment=True)
    test_dataset = Baby_Cry_Dataset(X_test, y_test, config)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader


# 使用示例
if __name__ == "__main__":
    #DATA_ROOT = "D:/pycharmproject/baby_cry_classify/baby_cry_data/train/"
    DATA_ROOT = "D:/pycharmproject/baby_cry_classify/baby_cry_data/train/"
    train_loader, test_loader = prepare_datasets(DATA_ROOT)