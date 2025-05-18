# src/evaluation.py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import torchaudio

from sklearn.metrics import roc_curve


class Evaluator:
    """
    Handles the evaluation of the speaker verification model.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _get_embeddings(self, file_paths, data_path, dataset_preprocessor):
        """Extracts embeddings for a list of unique audio files."""
        embeddings = {}
        unique_files = sorted(list(set(file_paths)))

        print("Extracting embeddings for evaluation...")
        for file_path in tqdm(unique_files):
            full_audio_path = os.path.join(data_path, file_path)

            log_mel_spec, processed_len = dataset_preprocessor.__getitem__(full_audio_path)

            if log_mel_spec is None:
                print(f"Warning: Skipping {full_audio_path} due to preprocessing issues (e.g., too short).")
                continue  # Skip this file if preprocessing failed

            model_input = log_mel_spec.unsqueeze(0).to(self.device)  # Shape becomes (1, 80, sequence_length)

            with torch.no_grad():
                embedding = self.model(model_input)
                embeddings[file_path] = embedding.squeeze(0)  # Remove batch dim before storing
        return embeddings

    def _calculate_eer(self, y_true, y_score):
        """Calculates the Equal Error Rate."""
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        # brentq expects function, lower bound, upper bound
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100

    def _calculate_mindcf(self, y_true, y_score, p_target=0.01, c_miss=1, c_fa=1):
        """
        Calculates the Minimum Detection Cost Function (minDCF).
        Parameters are set as in the paper.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr

        min_dcf = float("inf")
        for i in range(len(thresholds)):
            dcf = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if dcf < min_dcf:
                min_dcf = dcf
        return min_dcf

    def evaluate(self, trial_list_path, data_path, dataset_preprocessor):
        """
        Performs evaluation on a given trial list.

        Args:
            trial_list_path (str): Path to the trial list file (e.g., vox1_O_cleaned.txt).
            data_path (str): Path to the directory with test audio files.
            dataset_preprocessor: An object to preprocess audio, like an instance of AudioDataset.

        Returns:
            (float, float): The calculated EER and minDCF.
        """
        # 1. Read trial list and get unique files
        trials = np.loadtxt(trial_list_path, dtype=str)
        labels, files1, files2 = trials[:, 0], trials[:, 1], trials[:, 2]
        labels = [int(label) for label in labels]
        unique_files = np.unique(np.concatenate((files1, files2)))

        # 2. Extract embeddings for all unique files
        embeddings = self._get_embeddings(unique_files, data_path, dataset_preprocessor)

        # 3. Calculate scores for all pairs
        scores = []
        print(f"Calculating scores for {len(files1)} trials...")
        for f1, f2 in tqdm(zip(files1, files2)):
            emb1 = embeddings[f1]
            emb2 = embeddings[f2]
            score = F.cosine_similarity(emb1, emb2, dim=0).item()
            scores.append(score)

        # 4. Calculate metrics
        eer = self._calculate_eer(labels, scores)
        min_dcf = self._calculate_mindcf(labels, scores)

        print(f"Results for {os.path.basename(trial_list_path)}: EER = {eer:.3f}%, minDCF = {min_dcf:.4f}")
        return eer, min_dcf