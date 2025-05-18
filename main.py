#main.py

import torch
from torch.utils.data import DataLoader
from src.dataset import AudioDataset
from src.model import EcapaTdnn
from src.loss import AAMSoftmax
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.reporter import ReportGenerator
import os
import torchaudio


def main():
    TRAIN_DATA_PATH = './data/split_from_test/train_set'
    TEST_DATA_PATH = './data/split_from_test'
    TRIAL_LIST_PATH = './data/split_from_test/custom_trials.txt'
    MODEL_SAVE_PATH = 'model.pth'

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    TOTAL_EPOCHS = 50
    WARMUP_EPOCHS = 3
    EMB_DIM = 192
    CHANNELS = 1024

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")


    # --- Data Loading for Training ---
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: Training data path not found at '{TRAIN_DATA_PATH}'. Please run 'split_data.py' first.")
        return

    train_dataset = AudioDataset(data_path=TRAIN_DATA_PATH, for_training=True)

    if len(train_dataset) == 0:
        print(f"Error: No data found in '{TRAIN_DATA_PATH}'.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True  # pin_memory is not useful on MPS/CPU
    )
    num_classes = len(train_dataset.speaker_labels)
    print(f"Found {num_classes} speakers for training in the subset.")

    # ... (Rest of the file is the same as before)
    # --- Model and Loss Initialization ---
    model = EcapaTdnn(in_channels=80, channels=CHANNELS, emb_dim=EMB_DIM)
    loss_fn = AAMSoftmax(in_features=EMB_DIM, num_classes=num_classes, margin=0.2, scale=32)

    # --- Training ---
    trainer = Trainer(
        model=model, loss_fn=loss_fn, train_loader=train_loader, device=device,
        total_epochs=TOTAL_EPOCHS, warmup_epochs=WARMUP_EPOCHS, num_classes=num_classes
    )

    print("\n--- Starting Training on the Subset ---")
    trainer.train()
    print("\nTraining finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Evaluation ---
    print("\n--- Starting Evaluation on the Custom Test Set ---")

    if not os.path.exists(TRIAL_LIST_PATH):
        print(f"Error: Trial list not found at '{TRIAL_LIST_PATH}'. Please run 'generate_trials.py' first.")
        return

    eval_model = EcapaTdnn(in_channels=80, channels=CHANNELS, emb_dim=EMB_DIM)
    eval_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    evaluator = Evaluator(model=eval_model, device=device)
    eval_preprocessor = AudioDataset(data_path=None, for_training=False)

    report_generator = ReportGenerator(training_dataset_name="Custom Split")

    eer, mindcf = evaluator.evaluate(
        trial_list_path=TRIAL_LIST_PATH,
        data_path=TEST_DATA_PATH,
        dataset_preprocessor=eval_preprocessor
    )
    report_generator.add_result("Custom-Test", eer, mindcf)

    # --- Reporting ---
    report_generator.generate_report()


if __name__ == '__main__':
    main()