# src/esgmm_filter.py

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F

class EsgmmFilter:
    """
    Implements the ES-GMM filtering logic to identify noisy labels.
    """

    def __init__(self, device):
        self.device = device

    def identify_simple_samples(self, model_predictions, labels):
        """Identifies samples where the prediction matches the label."""
        predicted_labels = torch.argmax(model_predictions, dim=1)
        return predicted_labels == labels

    def calculate_embedding_centers(self, embeddings, labels, is_simple, num_classes, model_predictions):
        """
        Calculates the embedding center for each speaker using simple samples.
        See paper Section II-B-2.
        """
        centers = torch.zeros(num_classes, embeddings.size(1), device=self.device)
        counts = torch.zeros(num_classes, device=self.device)

        # Use simple samples to compute centers
        simple_embeddings = embeddings[is_simple]
        simple_labels = labels[is_simple]

        for i in range(num_classes):
            class_embeddings = simple_embeddings[simple_labels == i]
            if class_embeddings.size(0) > 0:
                centers[i] = class_embeddings.mean(dim=0)
                counts[i] = class_embeddings.size(0)

        # Handle speakers with no simple samples
        for i in range(num_classes):
            if counts[i] == 0:
                speaker_indices = (labels == i).nonzero(as_tuple=True)[0]
                if len(speaker_indices) > 0:
                    # Use the sample with the highest prediction score for this class
                    class_preds = model_predictions[speaker_indices][:, i]
                    best_sample_idx = speaker_indices[torch.argmax(class_preds)]
                    centers[i] = embeddings[best_sample_idx]

        return F.normalize(centers, p=2, dim=1)

    def gmm_binary_classification(self, similarities):
        """
        Classifies similarities into clean and noisy using a GMM.
        See paper Section II-B-3.
        """
        similarities_np = similarities.cpu().numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(similarities_np)

        # The clean cluster has the higher mean similarity
        clean_cluster_idx = np.argmax(gmm.means_)
        predictions = gmm.predict(similarities_np)

        is_clean_by_gmm = torch.tensor(predictions == clean_cluster_idx, device=self.device)
        return is_clean_by_gmm

    def secondary_filtering(self, embeddings, labels, is_noisy_by_gmm, centers):
        """
        Recalls hard samples that were misclassified as noisy by GMM.
        See paper Section II-B-4 and Equation 3.
        """
        recalled_indices = []
        noisy_indices = is_noisy_by_gmm.nonzero(as_tuple=True)[0]

        for idx in noisy_indices:
            emb = embeddings[idx]
            true_label = labels[idx]

            # Cosine similarity of the sample with its own speaker center
            sim_with_true_center = F.cosine_similarity(emb.unsqueeze(0), centers[true_label].unsqueeze(0))

            # Cosine similarity with all other speaker centers
            sim_with_other_centers = F.cosine_similarity(emb.unsqueeze(0), centers)

            # Check if the similarity with the true center is the highest
            if sim_with_true_center >= torch.max(sim_with_other_centers):
                recalled_indices.append(idx)

        return torch.tensor(recalled_indices, device=self.device)

    def filter_labels(self, embeddings, labels, model_predictions, num_classes):
        """
        Main function to perform the full ES-GMM filtering process.
        Returns a boolean mask where True indicates a clean sample.
        """
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 1: Identify Simple Samples
        is_simple = self.identify_simple_samples(model_predictions, labels)

        # Step 2: Calculate Embedding Centers
        centers = self.calculate_embedding_centers(
            normalized_embeddings, labels, is_simple, num_classes, model_predictions
        )

        # Step 3: Calculate Cosine Similarities
        similarities = torch.zeros(len(labels), device=self.device)
        for i in range(len(labels)):
            similarities[i] = F.cosine_similarity(
                normalized_embeddings[i].unsqueeze(0), centers[labels[i]].unsqueeze(0)
            )

        # Step 4: GMM Binary Classification
        is_clean_by_gmm = self.gmm_binary_classification(similarities)

        # Step 5: Secondary Filtering
        is_noisy_by_gmm = ~is_clean_by_gmm
        recalled_indices = self.secondary_filtering(normalized_embeddings, labels, is_noisy_by_gmm, centers)

        final_clean_mask = is_clean_by_gmm.clone()
        final_clean_mask[recalled_indices] = True

        num_clean = final_clean_mask.sum().item()
        num_total = len(labels)
        print(f"ES-GMM Filtered: Kept {num_clean}/{num_total} samples ({num_clean / num_total:.2%}) for training.")

        return final_clean_mask