from functools import partial
import os
import tempfile
import torch
import numpy as np
import argparse
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model import CNN_TRX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random

from ray.tune import Tuner
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as raypickle

seed = 2

#setting up seeds
def set_random_seed(manualSeed):
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

# max seq_len possible for hyperparameter config as a result form HP1
def choose_seq_len(method, temp_set, query_per_class):
    data = [
        {"method": "resnet18", "temp_set": [2, 3], "query_per_class": 1, "seq_len": 14},
        {"method": "resnet18", "temp_set": [2, 3], "query_per_class": 2, "seq_len": 12},
        {"method": "resnet18", "temp_set": [2, 3], "query_per_class": 3, "seq_len": 11},
        {"method": "resnet18", "temp_set": [2, 3], "query_per_class": 4, "seq_len": 10},
        {"method": "resnet18", "temp_set": [2, 3], "query_per_class": 5, "seq_len": 9},
        {"method": "resnet18", "temp_set": [2], "query_per_class": 1, "seq_len": 17},
        {"method": "resnet18", "temp_set": [2], "query_per_class": 2, "seq_len": 14},
        {"method": "resnet18", "temp_set": [2], "query_per_class": 3, "seq_len": 12},
        {"method": "resnet18", "temp_set": [2], "query_per_class": 4, "seq_len": 11},
        {"method": "resnet18", "temp_set": [2], "query_per_class": 5, "seq_len": 10},
        {"method": "resnet34", "temp_set": [2, 3], "query_per_class": 1, "seq_len": 11},
        {"method": "resnet34", "temp_set": [2, 3], "query_per_class": 2, "seq_len": 9},
        {"method": "resnet34", "temp_set": [2, 3], "query_per_class": 3, "seq_len": 8},
        {"method": "resnet34", "temp_set": [2, 3], "query_per_class": 4, "seq_len": 7},
        {"method": "resnet34", "temp_set": [2], "query_per_class": 1, "seq_len": 11},
        {"method": "resnet34", "temp_set": [2], "query_per_class": 2, "seq_len": 10},
        {"method": "resnet34", "temp_set": [2], "query_per_class": 3, "seq_len": 8},
        {"method": "resnet34", "temp_set": [2], "query_per_class": 4, "seq_len": 7}
    ]
    
    for entry in data:
        if entry["method"] == method and entry["temp_set"] == temp_set and entry["query_per_class"] == query_per_class:
            return entry["seq_len"]
    
    return None  # or raise an error if no match is found
    

def main(max_iterations=20000, data_dir="/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets"):
    test_config = {
    	"lr": 0.001, #tune.loguniform(1e-4, 1e-1),
    	"seq_len": 7,    	
        "temp_set": "2,3",
    	"method": "resnet18",
    	"dataset": "sp",
        "tasks_per_batch": 8,
    	"training_iterations": 20,
    	"way": 5, #tune.grid_search([n for n in range(5,8)]),
    	"shot": 5, #tune.grid_search([n for n in range(1,11)]),
    	"query_per_class": 1,
    	"query_per_class_test": 1,
    	"test_iters": [1,2,3,4,5,10],
        "num_test_task": 2,
    	"num_workers":10, # tune.grid_search([n for n in range(0,13,4)]), 
    	"trans_linear_out_dim": 1152,
    	"Optimizer": "adam",
    	"trans_dropout": 0.1,
    	"img_size": 224,
    	"num_gpus": 1,
    	"split": 5,
    }

    config_HP2_R18 = {
    	"lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
    	"temp_set": tune.grid_search([ "2", "2,3"]),
    	"method": "resnet18",
    	"dataset": "sp",
        "tasks_per_batch": 16,
    	"training_iterations": 10002,
    	"way": 5, #tune.grid_search([n for n in range(5,8)]),
    	"shot": 5, #tune.grid_search([n for n in range(1,11)]),
    	"query_per_class": tune.grid_search([n for n in range(1,6)]),
    	"query_per_class_test": 1,
    	"test_iters": [i for i in range(100,10002,100)],
        "num_test_task": 100,
    	"num_workers": 10, # tune.grid_search([n for n in range(0,13,4)]), 
    	"trans_linear_out_dim": 1152,
    	"Optimizer": "adam",
    	"trans_dropout": 0.1,
    	"img_size": 224,
    	"num_gpus": 1,
    	"split": 6
    }

    config_HP2_R34 = {
    	"lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
    	"temp_set": tune.grid_search([ "2", "2,3"]),
    	"method": "resnet34",
    	"dataset": "sp",
        "tasks_per_batch": 16,
    	"training_iterations": 10002,
    	"way": 5, #tune.grid_search([n for n in range(5,8)]),
    	"shot": 5, #tune.grid_search([n for n in range(1,11)]),
    	"query_per_class": tune.grid_search([n for n in range(1,5)]),
    	"query_per_class_test": 1,
    	"test_iters": [i for i in range(100,10002,100)],
        "num_test_task": 100,
    	"num_workers": 10, # tune.grid_search([n for n in range(0,13,4)]), 
    	"trans_linear_out_dim": 1152,
    	"Optimizer": "adam",
    	"trans_dropout": 0.1,
    	"img_size": 224,
    	"num_gpus": 1,
    	"split": 6
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=20000,
        grace_period=1000,
        reduction_factor=4
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train_cifar, data_dir=data_dir)),
            resources={"cpu": 24, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=1
        ),

        # TODO: chose your storage path
        run_config=train.RunConfig(
            storage_path="/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/hyperparameter_Tuning2",
            checkpoint_config=train.CheckpointConfig(num_to_keep=5,checkpoint_score_attribute="val_accuracy"),
        ),

        # TODO: chose your config
        param_space=config_HP2_R34
    )
    results = tuner.fit()

    return

def preprocess_config(config):
    temp_set = str.split(config["temp_set"], ",")
    config["temp_set"] = [int(t) for t in temp_set]
    if "seq_len" not in config:
        seq_len = choose_seq_len(
            method=config["method"],
            temp_set=config["temp_set"],
            query_per_class=config["query_per_class"]
        )
        config["seq_len"] = seq_len
    return config

def train_cifar(config, data_dir=None):
    set_random_seed(seed)
    preprocess_config(config)
    args = ArgsObject(data_dir, config)
    gpu_device = 'cuda'
    device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
    model = init_model(args, device)
    if config["Optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["Optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = os.path.join(checkpoint_dir,"data.pkl")
            with open(data_path, "rb") as fp:
                checkpoint_state = raypickle.load(fp)
            start_iteration = checkpoint_state["iteration"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_iteration = 0

    video_loader = load_data(args, data_dir)

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tf_config) as session:
        tasks_per_batch = config["tasks_per_batch"] 
        test_iters = config["test_iters"]
        train_accuracies = []
        losses = []
        used_train_classes = set()
        total_iterations = config["training_iterations"]
        best_accuracy = 0
        best_it = 0
        lowest_val_loss = 1000
        i = start_iteration

        for task_dict in video_loader:
            if i >= total_iterations:
                break
            i += 1

            task_loss, task_accuracy = train_task(task_dict, model, config["tasks_per_batch"], device)
            used_train_classes.update(task_dict['real_target_labels_names'][0])
            train_accuracies.append(task_accuracy.detach().cpu().numpy())
            losses.append(task_loss.detach().cpu().numpy())
            
            if (i % tasks_per_batch == 0) or (i == (total_iterations - 1)):
                optimizer.step()
                optimizer.zero_grad()

            if (i in test_iters): #and (i + 1) != total_iterations:
                accuracy, confidence, val_loss, used_val_classes = test(model, video_loader, config["num_test_task"], device)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_it = i
                    lowest_val_loss = val_loss

                with tempfile.TemporaryDirectory(prefix="", dir="/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/tmp") as checkpoint_dir:
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    # check if model has top 5 performance : yes -> save it
                    #if len(best_models) < 5 or accuracy > min(best_models.values()):
                    checkpoint_data = {
                        "iteration": i,
                        "net_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    data_path = os.path.join(checkpoint_dir, "data.pkl")
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)
                        # Update the best models dictionary
                        #if len(best_models) >= 5:
                        #    worst_model = min(best_models, key=best_models.get)
                        #    worst_model_path = worst_model
                        #    os.remove(worst_model_path)
                        #    del best_models[worst_model]
#
                        #best_models[data_path] = accuracy
                    
                    train.report({
                        "val_accuracy": float(accuracy), "val_loss": float(val_loss), "confidence": float(confidence),
                        "train_accuracy": np.array(train_accuracies).mean(), 
                        "train_loss": np.array(losses).mean(),
                        "used_train_classes": used_train_classes,
                        "used_val_classes": used_val_classes
                        },
                        checkpoint=checkpoint,
                    )
                    used_train_classes.clear()
                    train_accuracies = []
                    losses = []
    
    train.report({"val_accuracy": float(best_accuracy), "val_loss": float(lowest_val_loss), "it_high_val_acc": best_it})
    print("Finished Training")
            


def test(model, video_loader, num_test_task, device):
    model.eval()
    with torch.no_grad():

        video_loader.dataset.train = False
        test_loss = 0
        accuracies = []
        used_val_classes = set()

        for i, task_dict in enumerate(video_loader):
            if i >= num_test_task:
                break

            used_val_classes.update(task_dict['real_target_labels_names'][0])
            context_images, target_images, context_labels, target_labels, _, _, support_n_frames ,target_n_frames = prepare_task(task_dict, device)
            model_dict = model(context_images, context_labels, target_images, support_n_frames, target_n_frames)
            target_logits = model_dict['logits']
            accuracy = aggregate_accuracy(target_logits, target_labels)
            accuracies.append(accuracy.item())
            test_loss += loss(target_logits, target_labels, device)
            del target_logits

        accuracy = np.array(accuracies).mean() * 100.0
        confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

        video_loader.dataset.train = True
    
    test_loss = test_loss/num_test_task
    model.train()

    return accuracy, confidence, test_loss, used_val_classes

def train_task(task_dict, model, task_per_batch, device):
    torch.set_grad_enabled(True)
    context_images, target_images, context_labels, target_labels, _, _, support_n_frames ,target_n_frames = prepare_task(task_dict, device)
    model_dict = model(context_images, context_labels, target_images, support_n_frames, target_n_frames)
    target_logits = model_dict['logits']
    task_loss = loss(target_logits, target_labels, device) / task_per_batch
    task_accuracy = aggregate_accuracy(target_logits, target_labels)

    task_loss.backward(retain_graph=False)

    return task_loss, task_accuracy

def prepare_task(task_dict, device, images_to_device = True):
    context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
    target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
    real_target_labels = task_dict['real_target_labels'][0]
    batch_class_list = task_dict['batch_class_list'][0]
    support_n_frames = task_dict['support_n_frames'][0]
    target_n_frames = task_dict['target_n_frames'][0]

    if images_to_device:
        context_images = context_images.to(device)
        target_images = target_images.to(device)
    context_labels = context_labels.to(device)
    target_labels = target_labels.type(torch.LongTensor).to(device)
    support_n_frames = support_n_frames.to(device)
    target_n_frames = target_n_frames.to(device)

    return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, support_n_frames, target_n_frames


def init_model(args, device):
    model = CNN_TRX(args).to(device)
    
    return model

def load_data(args, data_dir):
    vd = video_reader.VideoDataset(args)
    video_loader = torch.utils.data.DataLoader(vd, batch_size=1, num_workers=args.num_workers)
    
    return video_loader

class ArgsObject(object):
    def __init__(self, data_dir, config):
        self.path = os.path.join(
            data_dir, "data", "surgicalphasev1_Xx256")
        self.traintestlist = os.path.join(
            data_dir, "splits", "surgicalphasev1TrainTestlist") # TODO hier die Pfade noch besser machen
        self.dataset = config["dataset"]
        self.split = config["split"]
        self.way = config["way"]
        self.shot = config["shot"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.seq_len = config["seq_len"]
        self.img_size = config["img_size"]
        self.temp_set = config["temp_set"]
        self.debug_loader = False
        if config["method"] == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
        self.trans_linear_out_dim = config["trans_linear_out_dim"]

        self.way = config["way"]
        self.shot = config["shot"]
        self.query_per_class = config["query_per_class"]
        self.trans_dropout = config["trans_dropout"]
        self.seq_len = config["seq_len"]
        self.img_size = config["img_size"]
        self.method = config["method"]
        self.num_gpus = config["num_gpus"]
        self.temp_set = config["temp_set"]
        self.num_workers = config["num_workers"]

if __name__ == "__main__":
    main()