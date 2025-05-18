# src/reporter.py

import pandas as pd
from datetime import datetime


class ReportGenerator:
    """
    Generates and saves a results report in a format similar to the paper's tables.
    """

    def __init__(self, training_dataset_name, model_name="ES-GMM"):
        self.results = []
        self.training_dataset_name = training_dataset_name
        self.model_name = model_name

    def add_result(self, test_set_name, eer, mindcf):
        """
        Adds a single evaluation result.

        Args:
            test_set_name (str): Name of the test set (e.g., 'Vox-O').
            eer (float): Equal Error Rate (%).
            mindcf (float): Minimum Detection Cost Function.
        """
        self.results.append({
            'Test Set': test_set_name,
            'EER(%)': f"{eer:.3f}",
            'minDCF': f"{mindcf:.4f}"
        })

    def generate_report(self, file_path="results_report.txt"):
        """
        Formats the results into a string and saves to a file.
        """
        if not self.results:
            print("No results to report.")
            return

        df = pd.DataFrame(self.results)

        report_str = f"""
# Speaker Verification Performance Report
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Training Dataset: {self.training_dataset_name}
- Method: {self.model_name}

## Performance on VoxCeleb1 Test Sets

"""
        # Create a combined table structure
        header = f"| {'Method':<15} |"
        separator = f"| {'-' * 15} |"
        method_row = f"| {self.model_name:<15} |"

        for _, row in df.iterrows():
            test_set = row['Test Set']
            header += f" {test_set + ' EER(%)':<15} | {test_set + ' minDCF':<15} |"
            separator += f" {'-' * 15} | {'-' * 15} |"
            method_row += f" {row['EER(%)']:<15} | {row['minDCF']:<15} |"

        report_str += header + "\n"
        report_str += separator + "\n"
        report_str += method_row + "\n"

        print("--- FINAL REPORT ---")
        print(report_str)

        with open(file_path, "w") as f:
            f.write(report_str)

        print(f"\nReport saved to {file_path}")