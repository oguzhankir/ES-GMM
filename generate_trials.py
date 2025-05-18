#generate_trials.py

import os
import random
from itertools import combinations


def create_trial_list(test_set_path, output_file_path):
    """
    Generates a trial list for speaker verification evaluation.

    The list will contain genuine pairs (label 1) and imposter pairs (label 0).

    Args:
        test_set_path (str): Path to the test set directory, which contains
                             subdirectories for each speaker.
        output_file_path (str): Path to save the generated trial list txt file.
    """
    print("--- Generating Custom Trial List ---")
    if not os.path.exists(test_set_path):
        print(f"Error: Test set directory not found at '{test_set_path}'")
        return

    # 1. Discover all speakers and their utterances
    speaker_files = {}
    print(f"Scanning directory: {test_set_path}")
    found_speakers_count = 0
    found_audio_count = 0

    for speaker_id in os.listdir(test_set_path):
        speaker_dir = os.path.join(test_set_path, speaker_id)
        if os.path.isdir(speaker_dir):
            found_speakers_count += 1
            speaker_files[speaker_id] = []
            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if file.endswith(".wav")  or file.endswith(
                            ".flac"):  # Include other common audio formats
                        full_path = os.path.join(root, file)
                        base_path_for_rel = os.path.dirname(test_set_path)
                        relative_path = os.path.relpath(full_path, base_path_for_rel)

                        speaker_files[speaker_id].append(relative_path)
                        found_audio_count += 1

    print(f"Found {found_speakers_count} speaker directories.")
    print(f"Found {found_audio_count} total audio files (.wav, .m4a, .flac) across all speakers.")

    if not speaker_files:
        print("No speakers or audio files found. Please check your `test_set_path` and audio file extensions.")
        return

    speakers_with_enough_files = {
        spk_id: files for spk_id, files in speaker_files.items() if len(files) >= 2
    }
    print(f"Speakers with at least 2 audio files for genuine pairs: {len(speakers_with_enough_files)}")

    trials = []

    # 2. Generate Genuine Trials (label 1)
    for speaker_id, files in speakers_with_enough_files.items():
        # A speaker must have at least 2 files to create a genuine pair
        for pair in combinations(files, 2):
            trials.append(f"1 {pair[0]} {pair[1]}")

    num_genuine_trials = len(trials)
    print(f"Generated {num_genuine_trials} genuine trials.")

    # 3. Generate Imposter Trials (label 0)
    speaker_list = list(speaker_files.keys())
    num_imposter_trials = 0

    if len(speaker_list) < 2:
        print("Not enough unique speakers (need at least 2) to generate imposter trials.")
    else:
        # We will generate a similar number of imposter trials as genuine trials, up to a limit
        target_imposter_count = max(num_genuine_trials, 500)  # Ensure a reasonable number even if genuine is low

        # To avoid infinite loops in small datasets, cap the attempts
        max_attempts = target_imposter_count * 2
        attempts = 0

        while num_imposter_trials < target_imposter_count and attempts < max_attempts:
            attempts += 1
            # Select two different speakers randomly
            spk1_id, spk2_id = random.sample(speaker_list, 2)

            # Ensure both speakers have at least one file
            if speaker_files[spk1_id] and speaker_files[spk2_id]:
                # Select one random file from each speaker
                file1 = random.choice(speaker_files[spk1_id])
                file2 = random.choice(speaker_files[spk2_id])

                trials.append(f"0 {file1} {file2}")
                num_imposter_trials += 1
            else:
                # print(f"Skipping imposter trial: one of the speakers ({spk1_id}, {spk2_id}) has no files.") # Debug print
                pass  # Continue to next attempt

    print(f"Generated {num_imposter_trials} imposter trials.")

    # 4. Shuffle and write to file
    if not trials:
        print("No trials (genuine or imposter) were generated. The output file will be empty.")
        with open(output_file_path, 'w') as f:
            pass  # Create empty file
        return

    random.shuffle(trials)

    with open(output_file_path, 'w') as f:
        for trial in trials:
            f.write(f"{trial}\n")

    print(f"\n--- Trial list successfully saved to '{output_file_path}' ---")


if __name__ == '__main__':
    # --- Configuration ---
    # Path to the test set we created with split_data.py
    CUSTOM_TEST_SET_PATH = 'data/split_from_test/test_set'

    # Path where the output trial list will be saved
    OUTPUT_TRIAL_FILE = 'data/split_from_test/custom_trials.txt'

    create_trial_list(CUSTOM_TEST_SET_PATH, OUTPUT_TRIAL_FILE)