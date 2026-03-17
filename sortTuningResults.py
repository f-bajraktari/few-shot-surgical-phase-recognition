from ray.tune import ExperimentAnalysis
import pandas as pd
import os
import shutil
import re
from collections import defaultdict

def read_save_sort_raytune_data(results_dir="C:\\Users\\roibl\\Downloads\\hyperparameter_Tuning\\train_cifar_2024-05-19_13-20-47",
                                destination_dir="C:\\Users\\roibl\\Downloads\\hyperparameter_Tuning\\copied_logs"):

    # Load the analysis object
    analysis = ExperimentAnalysis(results_dir)

    # Retrieve all trial dataframes
    df = analysis.dataframe()

    config_columns = [col for col in df.columns if col.startswith('config/') or col.startswith('logdir')]

    # Create a new DataFrame with the filtered columns
    config_df = df[config_columns]

    # Ensure 'method' and 'temp_set' are strings, convert if necessary
    config_df['config/method'] = config_df['config/method'].astype(str)
    config_df['config/temp_set'] = config_df['config/temp_set'].astype(str)

    # Get the index of the rows with the maximum seq_len for each group
    idx = config_df.groupby(['config/method', 'config/temp_set', 'config/query_per_class'])['config/seq_len'].idxmax()

    # Select the rows with the maximum seq_len
    max_seq_len_df = config_df.loc[idx]

    # Ensure there are no duplicate entries
    # Drop duplicates by selecting relevant columns and avoiding unhashable types
    max_seq_len_df = max_seq_len_df.drop_duplicates(subset=['config/method', 'config/temp_set', 'config/query_per_class', 'config/seq_len', 'logdir'])

    # Sort the DataFrame by the 'method' column
    max_seq_len_df = max_seq_len_df.sort_values(by=['config/method', 'config/temp_set'])

    # Display the result
    print(max_seq_len_df)

    # Optionally, save the successful trials to a CSV file
    max_seq_len_df.to_csv("successful_trials.csv", index=False)

    # Copy the directories listed in the 'logdir' column to the destination directory
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for log_id in max_seq_len_df['logdir']:
        # Find the full directory name based on the log_id
        full_logdir = None
        for dir_name in os.listdir(results_dir):
            if log_id in dir_name:
                full_logdir = os.path.join(results_dir, dir_name)
                break

        if full_logdir and os.path.isdir(full_logdir):
            destination_path = os.path.join(destination_dir, os.path.basename(full_logdir))
            shutil.copytree(full_logdir, destination_path)
    #
    ## Access the best trial based on a specific metric
    #best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    #print("Best trial:")
    #print(best_trial)
    #
    ## Access the best config
    #best_config = analysis.get_best_config(metric="val_loss", mode="min", scope="all")
    #print("Best config:")
    #print(best_config)
    #
    ## Access checkpoints from successful trials
    #for trial in successful_trials['logdir']:
    #    checkpoint_path = analysis.get_best_checkpoint(trial, metric="val_loss", mode="min")
    #    print(f"Best checkpoint for trial {trial}: {checkpoint_path}")

def remove_duplicate_runs(directory):
    count_deleted_folders = 0
    # Regex, um den relevanten Teil des Ordnernamens zu extrahieren
    pattern = re.compile(r'^[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')

    # Dictionary, um die Ordner nach ihrem relevanten Namen zu gruppieren
    folders = defaultdict(list)

    # Alle Ordner im Verzeichnis durchlaufen
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            key = match.group(1)
            folders[key].append(folder_name)

    # Ordner löschen, die sich nur im Timestamp und Prefix unterscheiden
    for key, folder_list in folders.items():
        # Behalte einen Ordner und lösche den Rest
        for folder_name in folder_list[1:]:
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                count_deleted_folders += 1
                #print(f"Lösche Ordner: {folder_path}")
                shutil.rmtree(folder_path)  # Verwenden Sie shutil.rmtree(folder_path) für nicht-leere Ordner
    
    print(f"Number of runs: {len(folders.items())}, deleted_folders: {count_deleted_folders}")

if __name__ == "__main__":
    read_save_sort_raytune_data()