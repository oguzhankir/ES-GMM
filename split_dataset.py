#split_dataset.py

import os
import shutil
import random

def split_dataset(source_dir, base_dest_dir, train_ratio=0.8):
    """
    Splits speaker directories from a source folder into training and testing sets.

    Args:
        source_dir (str): The path to the directory containing all speaker folders.
        base_dest_dir (str): The base path where 'train' and 'test' subdirectories will be created.
        train_ratio (float): The proportion of speakers to be used for the training set.
    """
    print(f"--- Starting to split data from '{source_dir}' ---")

    try:
        speaker_ids = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not speaker_ids:
            print(f"Error: No speaker directories found in '{source_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    # Shuffle the speakers randomly
    random.shuffle(speaker_ids)

    # Calculate the split index
    split_index = int(len(speaker_ids) * train_ratio)

    # Divide speaker IDs into training and testing sets
    train_speakers = speaker_ids[:split_index]
    test_speakers = speaker_ids[split_index:]

    # Define destination paths
    train_dest = os.path.join(base_dest_dir, 'train_set')
    test_dest = os.path.join(base_dest_dir, 'test_set')

    # Clean up old directories and create new ones
    for path in [train_dest, test_dest]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Function to copy speaker directories
    def copy_speakers(speaker_list, dest_path):
        for speaker_id in speaker_list:
            source_path = os.path.join(source_dir, speaker_id)
            shutil.copytree(source_path, os.path.join(dest_path, speaker_id))
        print(f"Copied {len(speaker_list)} speakers to '{dest_path}'")

    # Copy the files
    copy_speakers(train_speakers, train_dest)
    copy_speakers(test_speakers, test_dest)

    print("\n--- Splitting complete! ---")
    print(f"Total speakers: {len(speaker_ids)}")
    print(f"Training speakers: {len(train_speakers)}")
    print(f"Testing speakers: {len(test_speakers)}")

if __name__ == '__main__':
    SOURCE_DATA_PATH = 'data/aac'

    DESTINATION_PATH = 'data/split_from_test'

    split_dataset(SOURCE_DATA_PATH, DESTINATION_PATH)