# src/dataset.py (Revised)

import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class AudioDataset(Dataset):
    """
    A PyTorch Dataset for handling audio files, specifically for speaker verification.
    It supports loading audio, extracting features, and optionally handling labels for training.
    """

    def __init__(self, data_path, for_training=True, sample_rate=16000, n_mels=80,
                 segment_len_seconds=3.0, min_len_seconds=1.0):
        self.data_path = data_path
        self.for_training = for_training
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.segment_len_seconds = segment_len_seconds
        self.min_len_seconds = min_len_seconds

        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(self.sample_rate * 0.025),
            hop_length=int(self.sample_rate * 0.01),
            n_mels=self.n_mels
        )
        self.amplitude_to_db_transform = AmplitudeToDB()

        self.audio_files = []
        self.speaker_labels = {}
        self.label_to_speaker = []

        if self.for_training and data_path:
            self._load_training_data()

    def _load_training_data(self):
        speaker_id_counter = 0
        for speaker_id in sorted(os.listdir(self.data_path)):
            speaker_dir = os.path.join(self.data_path, speaker_id)
            if os.path.isdir(speaker_dir):
                if speaker_id not in self.speaker_labels:
                    self.speaker_labels[speaker_id] = speaker_id_counter
                    self.label_to_speaker.append(speaker_id)
                    speaker_id_counter += 1

                for root, _, files in os.walk(speaker_dir):
                    for file in files:
                        if file.endswith((".wav", ".flac")):  # Ensure .m4a is excluded if you've converted
                            file_path = os.path.join(root, file)
                            self.audio_files.append({
                                'path': file_path,
                                'label': self.speaker_labels[speaker_id]
                            })
        if not self.audio_files:
            print(f"Warning: No audio files found in {self.data_path}. Please check path and file extensions.")

    def _preprocess_audio(self, audio_path):
        """
        Loads audio, resamples, extracts a segment, and computes mel spectrogram.
        Ensures the output mel spectrogram is (n_mels, num_frames).
        """
        try:
            waveform, sr = torchaudio.load(audio_path)

            if waveform.dim() > 2:  # More than 2 dimensions (e.g., [1, 1, samples])
                waveform = waveform.squeeze()  # Remove all dimensions of size 1
            if waveform.dim() == 1:  # If it became (samples,), add channel dim for MelSpectrogram
                waveform = waveform.unsqueeze(0)  # Becomes (1, samples)
            if waveform.shape[0] > 1:  # If it's stereo (2, samples), take first channel
                waveform = waveform[0, :].unsqueeze(0)  # Ensure it's (1, samples)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            min_samples = int(self.min_len_seconds * self.sample_rate)
            if waveform.shape[1] < min_samples:
                return None, None

            if self.for_training:
                segment_samples = int(self.segment_len_seconds * self.sample_rate)
                if waveform.shape[1] > segment_samples:
                    start_idx = random.randint(0, waveform.shape[1] - segment_samples)
                    waveform = waveform[:, start_idx: start_idx + segment_samples]
                else:
                    pad_needed = segment_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_needed), 'constant', 0)

            mel_spec = self.mel_spectrogram_transform(waveform)
            log_mel_spec = self.amplitude_to_db_transform(mel_spec)

            if log_mel_spec.dim() > 2:
                log_mel_spec = log_mel_spec.squeeze(0)  # Remove the first dimension if it's 1

            return log_mel_spec, waveform.shape[1]
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return None, None

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if self.for_training:
            item = self.audio_files[idx]
            log_mel_spec, processed_len = self._preprocess_audio(item['path'])
            if log_mel_spec is None:
                # If preprocessing failed or audio was too short, try another random item
                return self.__getitem__(random.randint(0, len(self) - 1))
            return log_mel_spec, item['label']
        else:
            if isinstance(idx, str):
                log_mel_spec, processed_len = self._preprocess_audio(idx)
                if log_mel_spec is None:
                    raise RuntimeError(f"Failed to preprocess evaluation file: {idx}")
                return log_mel_spec, processed_len
            else:
                raise IndexError(f"AudioDataset in evaluation mode expects a file path as index, got {type(idx)}")
