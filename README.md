Project Team
------------

This project was developed by the following team members:

*   **OĞUZHAN KIR** (Student ID: 528231089)
*   **ÖYKÜ SU BAŞARAN** (Student ID: 504241591)
*   **VEYSEL EMRE KÖSE** (Student ID: 504241582)

* * *

Table of Contents
-----------------

*   [Project Overview](#project-overview)
*   [Key Components](#key-components)
*   [Dataset Setup](#dataset-setup)
    *   [Original Dataset Used](#original-dataset-used)
    *   [Local Data Preparation](#local-data-preparation)
    *   [.m4a to .wav Conversion](#m4a-to-wav-conversion)
    *   [Generating Custom Trial List](#generating-custom-trial-list)
*   [Installation](#installation)
*   [Running the Project](#running-the-project)
*   [Results](#results)
*   [Troubleshooting](#troubleshooting)

* * *

Project Overview
----------------

This project focuses on addressing the challenge of noisy labels in large-scale speaker verification datasets. Noisy labels, inevitable in automatically collected data, can lead to deep neural networks overfitting incorrect information and degrading performance. The implemented solution, ES-GMM, proposes an efficient method to filter out these noisy labels during the training process.

The core mechanism of ES-GMM involves:

1.  **Warm-up Stage:** An initial training phase (e.g., 3 epochs) allows the model to gain a basic ability to distinguish speakers and learn general patterns before overfitting to noise.
2.  **Embedding Center Calculation:** For each speaker, a normalized embedding center is computed by averaging the embeddings of "simple samples" (samples where the model's prediction matches its label). A fallback mechanism ensures centers for speakers with no simple samples.
3.  **Cosine Similarity Calculation:** The cosine similarity between each audio embedding and its corresponding speaker's embedding center is calculated.
4.  **GMM Binary Classification:** A Gaussian Mixture Model (GMM) is used for unsupervised binary classification of these cosine similarities into "clean" (higher mean similarity) and "noisy" (lower mean similarity) categories. This dynamically learned threshold is more robust than fixed thresholds.
5.  **Secondary Filtering:** A "secondary filtering" step re-evaluates samples initially classified as noisy. If a sample's similarity with its true label's speaker center is higher than with any other speaker's center, it is recalled as a clean sample. This helps re-introduce "hard but clean" samples into training.
6.  **Iterative Training:** Only the samples identified as "clean" are used for backpropagation in the current epoch. This iterative process, reminiscent of curriculum learning, gradually filters noise and incorporates harder, correctly labeled samples over time.

Key Components
--------------

*   `src/dataset.py`: Manages the loading, preprocessing (resampling, segment extraction, Mel-spectrogram, dB scaling), and dimension handling of audio files. It maps speaker IDs to numerical labels and prepares data for both training and evaluation. It is configured to recognize `.wav` and `.flac` files.
*   `src/esgmm_filter.py`: Encapsulates the core ES-GMM algorithm. It handles the identification of simple samples, computation of speaker embedding centers, GMM-based classification of cosine similarities, and the secondary filtering step for recalling hard samples.
*   `src/evaluation.py`: Responsible for evaluating the speaker verification model's performance. It extracts embeddings, calculates cosine similarities between trial pairs, and computes standard speaker verification metrics: Equal Error Rate (EER) and Minimum Detection Cost Function (minDCF).
*   `src/loss.py`: Defines the Additive Angular Margin Softmax (AAM-Softmax) loss function, which is utilized for training the speaker verification model.
*   `src/model.py`: Implements the ECAPA-TDNN architecture, a widely used speaker embedding extractor in speaker verification. It processes log Mel-spectrogram features and outputs fixed-dimensional embeddings.
*   `src/reporter.py`: Generates a formatted report of the experimental results, specifically EER and minDCF on evaluation sets, structured similarly to the paper's tables.
*   `src/trainer.py`: Orchestrates the training loop, managing the model optimization with SGD, learning rate scheduling (warm-up followed by cosine annealing), and integrating the ES-GMM filtering logic for sample selection during training epochs.
*   `split_dataset.py`: A utility script that divides the raw dataset based on speaker IDs into separate training and testing directories (e.g., `train_set` and `test_set`).
*   `generate_trials.py`: A utility script to create evaluation trial lists (`.txt` files) by generating genuine (same speaker) and imposter (different speakers) pairs from a given test set.

Dataset Setup
-------------

### Original Dataset Used

The project utilizes data derived from the VoxCeleb dataset, specifically focusing on the test portion of VoxCeleb2. The full VoxCeleb dataset is exceptionally large (over 300GB compressed). For this project, a more manageable subset from the VoxCeleb2 test set was used for local training and evaluation. The specific files downloaded were:

*   `vox2_test_mp4.zip`
*   `vox2_test_aac.zip`
*   `vox2_test_text.zip`

These compressed files sum up to over 10GB. After extraction, the `data` folder of the project occupies approximately 18GB of disk space.

The `data` folder initially contains `aac`, `mp4`, `split_from_test`, `txt 2`, `txt 3` directories/files after extracting the downloaded archives.

    data/
    ├── aac/             # Contains speaker subdirectories with .m4a audio files
    ├── mp4/
    ├── split_from_test/ # Will contain train_set and test_set after splitting
    ├── txt 2/
    └── txt 3/
    

### Local Data Preparation

Before running the main training and evaluation script, the raw data needs to be prepared:

1.  **Extract Downloaded Archives:** Ensure you have extracted `vox2_test_aac.zip` (and other related zips) into the `data/` directory of your project. The `aac` directory should contain speaker subdirectories (e.g., `data/aac/id00001/`).
2.  **Split Dataset:** Use the `split_dataset.py` script to partition the `data/aac` directory into a training set and a test set based on speaker IDs. This creates `data/split_from_test/train_set` and `data/split_from_test/test_set`.
    
        python split_dataset.py
        
    
    **Note:** This script copies speaker directories. Ensure you have sufficient disk space.
    
3.  **Convert .m4a to .wav:** The extracted AAC files are often in `.m4a` format. Since `soundfile` (used by `torchaudio`) may not natively support `.m4a`, converting these to `.wav` is crucial to prevent "Format not recognised" errors during audio loading.
    1.  **Install FFmpeg:** If not already installed, ensure FFmpeg is on your system.
        
            # For macOS with Homebrew
            brew install ffmpeg
            
        
    2.  **Navigate to Project Root:** Open your terminal and change your directory to the project's root folder (e.g., `ES-GMM`).
        
            cd /Users/oguz/Desktop/machine\ learning/Term\ Project/ES-GMM
            
        
    3.  **Run Conversion Command:** Execute the following shell command. This command recursively finds all `.m4a` files within `data/split_from_test/test_set` and converts them to `.wav` files in their respective locations.
        
            find "./data/split_from_test/test_set" -name "*.m4a" -exec sh -c 'ffmpeg -i "$0" "${0%.m4a}.wav"' {} \;
            
        
        **Important:** This process can be time-consuming for large datasets. Ensure the command runs to completion without interruption.
        
4.  **Remove Deprecated Torchaudio Backend Calls:** Although not causing direct crashes anymore, the warnings about `torchaudio._backend.set_audio_backend` being deprecated are present in your console output. It's good practice to remove these lines from your code.
    *   In `src/dataset.py`: Remove `torchaudio.set_audio_backend("ffmpeg")` (line 12).
    *   In `main.py`: Remove the `try-except` block for `torchaudio.set_audio_backend` (lines 14-18).
5.  **Generate Custom Trial List:** Create the evaluation trial list for your `test_set` using the `generate_trials.py` script. This will produce `data/split_from_test/custom_trials.txt`.
    
        python generate_trials.py
        
    

Installation
------------

Before running the project, ensure you have the necessary Python packages and system dependencies installed. It's highly recommended to use a virtual environment.

1.  **Create and Activate Virtual Environment:**
    
        python3 -m venv .venv
        source .venv/bin/activate  # On Linux/macOS
        # .venv\Scripts\activate  # On Windows
        
    
2.  **Install Dependencies:** All required Python packages are listed in `requirements.txt`.
    
        pip install -r requirements.txt
        
    
    The `requirements.txt` includes:
    
    *   `asteroid-filterbanks==0.4.0`
    *   `av==14.4.0`
    *   `cffi==1.17.1`
    *   `ffmpeg==1.4`
    *   `filelock==3.18.0`
    *   `fsspec==2025.5.1`
    *   `Jinja2==3.1.6`
    *   `joblib==1.5.1`
    *   `MarkupSafe==3.0.2`
    *   `mpmath==1.3.0`
    *   `networkx==3.5`
    *   `numpy==2.2.6`
    *   `pandas==2.3.0`
    *   `pycparser==2.22`
    *   `pydub==0.25.1`
    *   `python-dateutil==2.9.0.post0`
    *   `pytz==2025.2`
    *   `PyYAML==6.0.2`
    *   `scikit-learn==1.7.0`
    *   `scipy==1.15.3`
    *   `six==1.17.0`
    *   `soundfile==0.13.1`
    *   `sympy==1.14.0`
    *   `threadpoolctl==3.6.0`
    *   `torch==2.7.1`
    *   `torchaudio==2.7.1`
    *   `tqdm==4.67.1`
    *   `typing_extensions==4.14.0`
    *   `tzdata==2025.2`

Running the Project
-------------------

After completing all the [Dataset Setup](#dataset-setup) steps, you can run the main training and evaluation script.

The `main.py` script is configured to train an ECAPA-TDNN model with the ES-GMM filter (after a warm-up period) on your `data/split_from_test/train_set` and then evaluate it on `data/split_from_test/custom_trials.txt`.

Execute the script from your project's root directory:

    python main.py
    

**Runtime Information:**

*   The script automatically detects and utilizes an Apple Silicon (MPS) GPU, NVIDIA CUDA GPU, or falls back to CPU for computations.
*   Model training on the current small subset of data (approx. 18GB) with an RTX 4070 GPU (12GB VRAM) took approximately **14 hours**. Expect similar or longer durations depending on your hardware.

Results
-------

Upon completion of the `main.py` script, a report named `results_report.txt` will be generated in your project's root directory. This report summarizes the speaker verification performance based on Equal Error Rate (EER) and Minimum Detection Cost Function (minDCF).

### Sample Results (Custom Split)

The following are sample results obtained from a run using the current code, training on the locally split training set and evaluating on the custom test set. These results are indicative of the system's performance on a smaller, potentially less diverse subset of the original VoxCeleb data. For direct comparison with the paper's benchmarks (which often use larger datasets and specific noise injection scenarios), further data preparation and experimental configurations would be required.

    # Speaker Verification Performance Report
    - Date: 2025-06-11 22:57:32
    - Training Dataset: Custom Split
    - Method: ES-GMM
    
    ## Performance on VoxCeleb1 Test Sets
    
    | Method          | Custom-Test EER(%) | Custom-Test minDCF |
    | --------------- | --------------- | --------------- |
    | ES-GMM          | 43.703          | 0.0100          |
    

The result above shows an EER of 43.703% and a minDCF of 0.0100 on the "Custom-Test" set. These figures serve as a baseline for the implemented system on the specific data split used in this project.

### Interpreting EER and minDCF

EER (Equal Error Rate) and minDCF (Minimum Detection Cost Function) are standard metrics for speaker verification systems:

*   **EER (%):** Represents the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). A lower EER indicates a more accurate system.
*   **minDCF:** A weighted sum of FAR and FRR, taking into account specific costs associated with misidentifications. A lower minDCF signifies better overall system performance. The paper's minDCF settings are $P\_{target}=0.01$ and $C\_{fa}=C\_{miss}=1$.

Troubleshooting
---------------

*   **`RuntimeError: Expected 2D or 3D input to conv1d, but got input of size: [1, 1, 80, 455]`:**
    
    This error indicates an extra channel dimension in the input to the convolutional layer. The provided `src/dataset.py` code already includes logic (`waveform.squeeze()` and `log_mel_spec.squeeze(0)`) to ensure the Mel-spectrogram tensor has the correct dimensions (`[num_mels, num_frames]`) before adding the batch dimension. Ensure your `src/dataset.py` is fully updated with these fixes.
    
*   **`soundfile.LibsndfileError: Error opening '...m4a': Format not recognised.`:**
    
    This error occurs when `torchaudio` attempts to load `.m4a` files using the `soundfile` backend, which does not support this format. The solution is to convert all your `.m4a` audio files to `.wav` format using the `ffmpeg` command provided in the [.m4a to .wav Conversion](#m4a-to-wav-conversion) section. After conversion, `generate_trials.py` should predominantly detect `.wav` files, which `soundfile` handles reliably.
    
*   **`Error: No speaker directories found in '...'` or `find: No such file or directory`:**
    
    These errors typically mean the specified data paths are incorrect or the directories are empty. Double-check that your downloaded dataset is extracted correctly into `data/aac` and that you're running `split_dataset.py` from the correct project root. When using `find` commands, ensure the full path is correct and properly quoted or escaped (e.g., `machine\ learning`).
    
*   **`Generated 0 genuine trials` or empty `custom_trials.txt`:**
    
    This indicates that `generate_trials.py` could not find enough suitable audio files (at least two per speaker) in the `data/split_from_test/test_set` directory to create evaluation pairs. Verify that the `.wav` conversion was successful and that the `test_set` contains populated speaker folders with `.wav` or `.flac` files.
    
*   **`UserWarning: torchaudio._backend.set_audio_backend has been deprecated.`:**
    
    This is a warning, not an error, indicating that the `torchaudio.set_audio_backend("ffmpeg")` call is no longer necessary or effective in newer `torchaudio` versions due to its dispatcher mechanism. The provided code has instructions to remove these lines in the [.m4a to .wav Conversion](#m4a-to-wav-conversion) section.