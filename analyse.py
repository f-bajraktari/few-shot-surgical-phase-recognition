import os

from ray import train, tune
from ray.tune.examples.mnist_pytorch import train_mnist
from ray.tune.analysis import ExperimentAnalysis
import pandas as pd
import shutil
import matplotlib.pyplot as plt

def load_experiment(experiment_path):
    # Load the experiment analysis
    analysis = ExperimentAnalysis(experiment_path)

    return analysis

def analyse_experiments(results_list):
    # List to store trial information
    trials_info = []
    
    for results in results_list:
        # Iterate over all trials
        for result in results:
            # Get trial configuration
            config = result.config
            metrics = result.metrics
    
            # # Fetch the best checkpoint for the current trial
            # best_checkpoint = analysis.get_best_checkpoint(trial, metric="val_accuracy", mode="max")
    
            # # Fetch the validation accuracy and checkpoint number of the best checkpoint
            # if best_checkpoint:
            #     # Load the best checkpoint data
            #     checkpoint_path = best_checkpoint.path
            #     checkpoint_number = None
                
            #     # Assuming checkpoints are saved with iteration number in their filename
            #     # This part depends on your specific checkpoint naming convention
            #     checkpoint_filename = os.path.basename(checkpoint_path)
            #     try:
            #         checkpoint_number = int(checkpoint_filename.split('_')[-1])
            #     except ValueError:
            #         checkpoint_number = None
                
            #     best_checkpoint_score = trial.metric_analysis["val_accuracy"]["max"]
            # else:
            #     best_checkpoint_score = None
            #     checkpoint_number = None

            # Collect relevant information
            trials_info.append({
                "trial_path": result.path,
                "method": config.get("method", None),
                "temp_set": config.get("temp_set", None),
                "query_per_class": config.get("query_per_class", None),
                "seq_len": config.get("seq_len", None),
                "lr": config.get("lr", None),
                "val_accuracy": metrics.get("val_accuracy", None),
                "iteration": metrics.get("it_high_val_acc", None)
            })

    # Convert the list of dictionaries to a DataFrame
    trials_df = pd.DataFrame(trials_info)
    
    trials_df = trials_df.dropna(subset=['val_accuracy'])

    # Sort the DataFrame by val_accuracy in descending order
    trials_df = trials_df.sort_values(by="val_accuracy", ascending=False)
    
    # Save the DataFrame to a CSV file
    #trials_df.to_csv("HP2_results.csv", index=False)

    
    # Print the DataFrame to verify
    print(trials_df)

    return trials_df

def copy_trials(output_dir, trials_path_list):
    # Copy the directories listed in the 'logdir' column to the destination directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for trial_path in trials_path_list:
        base_name = os.path.basename(trial_path)
        output_dir_full = os.path.join(output_dir, base_name)
        if not os.path.exists(output_dir_full):
            shutil.copytree(trial_path, output_dir_full)
            print(f"copied {trial_path}")

def plot_trials_metric(trials, metric, x_name, y_name):
    """
    Plots the specified metric for a list of trials.

    Args:
    trials (list): List of trial dataframes, each containing the metrics over epochs.
    metric (str): The metric to plot (e.g., "val_accuracy").
    """
    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Iterate through the trials and plot their specified metric
    for i, trial_result in enumerate(trials):
        metric_df = trial_result.metrics_dataframe
        # Check if the trial data contains the metric
        if metric not in metric_df:
            print(f"Metric '{metric}' not found in trial {i+1}. Skipping this trial.")
            continue
        
        metric_list = metric_df[metric].to_list()
        # metric_list = [i*100 for i in metric_list]
        epoch_list = metric_df["training_iteration"].to_list()
        epoch_list = [i*100 for i in epoch_list]
        # Plot the specified metric
        plt.plot(epoch_list, metric_list, label=f"Kombination {i+1}")

    # Customize the plot
    plt.xlabel(x_name, fontsize = 32)
    plt.ylabel(y_name, fontsize = 32)
    plt.legend(loc='lower right', fontsize = 20, ncol = 2)
    # Calculate x-axis tick positions
    x_ticks = range(0, 10001, 1000)  # Adjust as needed based on your epochs range
    
    plt.xticks(x_ticks, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    experiments_path = "G:\\Meine Ablage\\Studium\\Master\\Forschungsarbeit\\05_Data\\TRX\\hyperparameter_Tuning2"
    #output_dir = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/HP2_best_results"
    analysis_list = []
    for path in os.listdir(experiments_path):
        experiment_path = os.path.join(experiments_path, path)
        restored_tuner = tune.Tuner.restore(experiment_path, trainable='tune_with_parameters')
        result_grid = restored_tuner.get_results()
        analysis_list.append(result_grid)
    
    trials_df = analyse_experiments(analysis_list)
    # Assuming trials_df is your DataFrame
    first_10_trials = trials_df['trial_path'].iloc[:10].to_list()
    trial_result_list = [train.Result.from_path(path) for path in first_10_trials]
    plot_trials_metric(trial_result_list, "val_accuracy", "Epoche", "$acc_{val}$ in %")

    # Call the function with the first 10 entries
   # copy_trials(output_dir, first_10_trials)