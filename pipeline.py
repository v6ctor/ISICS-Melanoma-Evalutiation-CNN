from typing import Tuple, Union
import torch, torchvision
import logging
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch.nn as nn
import albumentations as A
import seaborn as sns

from classifier import Net

from filelock import FileLock
from datetime import datetime

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from torch.utils.tensorboard import SummaryWriter

from collections import Counter

from RandAugment import RandAugment
from albumentations.pytorch import ToTensorV2
from custom_transformations import AdvancedHairAugmentation, AlbumentationsTransform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

class Pipeline:
    def __init__(self, log_path: str, download_path: str, classes: Tuple[str]):
        self.train_set = None
        self.test_set = None

        self.classes = classes

        self.logger = None
        self.log_path = log_path

        self.download_path = download_path

        self.predictions = []
        self.test_truth = []
        self.training_loss = []
        self.validation_loss = []

        self.model_accuracies = []

        self.device = torch.device("cuda")  
      
    def init_debug_logger(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def reformat_data_folder(self, source: str) -> None:
        """
        Purpose: reads patient data from CSV files, organizes images into "malignant" and "benign" folders based on labels,
                 and moves images to their appropriate folders.
        Params:  (source) path to the directory where the image files will be moved and organized based on their classification
                 (malignant or benign)
        Returns: none
        """
        try:
            # get patient data
            patient_data = pd.read_csv('/home/xpeshla1/scratch/skin_lesion/ISIC_2020_Training_GroundTruth_v2.csv')

            # make malignant and benign folder in path
            os.makedirs(os.path.join(source, "malignant"), exist_ok=True)
            os.makedirs(os.path.join(source, "benign"), exist_ok=True)

            # ensure no duplicates
            patient_data.drop_duplicates(subset=["image_name"], inplace=True)

            # locate malignant and bengin 
            malignant = patient_data.loc[patient_data["benign_malignant"] == "malignant", "image_name"].values
            benign = patient_data.loc[patient_data["benign_malignant"] == "benign", "image_name"].values

            # check if malignant image from soure exists in destination, if not add it
            for image_name in malignant:
                destination = os.path.join(source + "malignant", image_name + ".jpg")
                if os.path.exists(destination):
                    continue
                source = os.path.join(source, image_name + ".jpg")
                if not os.path.exists(source):
                    continue
                 
                os.rename(source, destination)
            
            # check if benign image from souce exists in destination, if not add it
            for image_name in benign:
                destination = os.path.join(source + "benign", image_name + ".jpg")
                if os.path.exists(destination):
                    continue
                source = os.path.join(source, image_name + ".jpg")

                if not os.path.exists(source):
                    continue
                 
                os.rename(source, destination)

            # use 2018 data for validation
            validation_data = pd.read_csv("/scratch/xpeshla1/skin_lesion/validation_2018/ISIC2018_Task3_Validation_GroundTruth.csv")
            os.makedirs(os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018", "malignant"), exist_ok = True)
            os.makedirs(os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018", "benign"), exist_ok = True)

            # find malginant
            malignant = validation_data.loc[
                (validation_data["MEL"] == 1.0) | (validation_data["image"] == "malignant"), 
                "image"
            ].values
            
            # find benign
            benign = validation_data.loc[
                (validation_data["NV"] == 1.0) | (validation_data["BCC"] == 1.0) |
                (validation_data["AKIEC"] == 1.0) | (validation_data["BKL"] == 1.0) |
                (validation_data["DF"] == 1.0) | (validation_data["VASC"] == 1.0), 
                "image"
            ].values

            # checks images in malignant source and adds it to destination
            for image_name in malignant:
                destination = os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018/malignant", image_name + ".jpg")
                if os.path.exists(destination):
                    continue
                source = os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018", image_name + ".jpg")
                if not os.path.exists(source):
                    continue
                 
                os.rename(source, destination)
            # checks images in benign source and adds it to destination
            for image_name in benign:
                destination = os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018/benign", image_name + ".jpg")
                if os.path.exists(destination):
                    continue
                source = os.path.join("/scratch/xpeshla1/skin_lesion/validation_2018", image_name + ".jpg")

                if not os.path.exists(source):
                    continue
                os.rename(source, destination)

        except Exception as e:
            self.logger.debug(f"Failed to reformat data folders: {e}")

    def download_data(self, testing: bool = False) -> Union[Tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder], int]:
        """
        Purpose: downloads image datasets with optional transformations for testing and returns the datasets or -1 if an error occurs.
        Params: param testing: a boolean flag that indicates whether the data download is for testing purposes or not. If "testing" is set to True, a
                specific set of non-augmented transformations will be applied to the downloaded data. Otherwise, a different set of augmented transformations will, 
                defaults to False
        Return: returns either two ImageFolder objects (representing the training dataset and the validation
                dataset) or -1 if an exception occurs during the process of downloading the datasets.
        """
        try:
            # perform basic transformations on testing images to standardize size and normalize
            if testing:
                transformation = A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                    ToTensorV2()
                ], seed=42)
            else:
            # perform advance transformations on training images to reduce destracting features like air and angle
                transformation = A.Compose([
                    A.Resize(256, 256),
                    A.HorizontalFlip(p = 0.5),
                    A.VerticalFlip(p = 0.5),
                    A.SafeRotate(90.0),
                    AdvancedHairAugmentation(hairs = 6, p = 0.5),
                    A.CoarseDropout(max_holes = 8, max_height = 10, max_width = 10, p = 0.4),
                    A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])

            self.reformat_data_folder(self.download_path)

            albumentations_transform = AlbumentationsTransform(transformation)

            with FileLock(os.path.expanduser(f"~/{self.download_path}.lock")):

                dataset = torchvision.datasets.ImageFolder(root = self.download_path, transform = albumentations_transform)

                # seperate benign and malignant images
                benign_class_idx = dataset.class_to_idx.get("benign")
                benign_images = [item for item in dataset.samples if item[1] == benign_class_idx]
                malignant_images = [item for item in dataset.samples if item[1] != benign_class_idx]

                # reduce number of benign images down to 584 images to be equivalent to number of malignant images
                max_benign_count = 584
                if len(benign_images) > max_benign_count:
                    benign_images = benign_images[:max_benign_count]
                reduced_samples = benign_images + malignant_images
                dataset.samples = reduced_samples
                dataset.targets = [sample[1] for sample in reduced_samples]

                transformation = A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])

                albumentations_transform = AlbumentationsTransform(transformation)
                
                validation_set = torchvision.datasets.ImageFolder(root = "/scratch/xpeshla1/skin_lesion/validation_2018", transform = albumentations_transform)

                return dataset, validation_set
        except Exception as e:
            self.logger.debug(f"Failed to download datasets: {e}")

            return -1

    def split_data(self, dataset: torchvision.datasets.ImageFolder, seed: int = None) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        """
        Purpose: takes an ImageFolder dataset and splits it into training and testing subsets while maintaining class balance.
        Params: dataset, a dataset that this method splits into training and testing subsets,
                seed, sets the random seed for reproducibility when splitting the dataset into training and testing subsets.
        Return: a tuple containing two subsets of the input dataset: train_subset and test_subset.
        """

        try:
            # get data set size and split dataset into train and test 80/20 
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            labels = [dataset.samples[i][1] for i in indices]
            train_indices, test_indices = train_test_split(indices, test_size = 0.2, stratify = labels, random_state = seed)

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)

            # print(f"Number of samples in the train set: {len(train_subset)}")
            # print(f"Number of samples in the test set: {len(test_subset)}")

            # labels = [train_subset[i][1] for i in range(len(train_subset))]
            # label_counts = Counter(labels)
            # print(f"Class distribution for train set: {label_counts}")
            
            # labels = [test_subset[i][1] for i in range(len(test_subset))]
            # label_counts = Counter(labels)
            # print(f"Class distribution for test set: {label_counts}")

            return train_subset, test_subset

        except Exception as e:
            self.logger.debug(f"Failed to split dataset: {e}")

            return -1

    def split_subset_to_csv(self, dataset: torchvision.datasets.ImageFolder, subset: torch.utils.data.Subset) -> None:
        """
        Purpose: takes a subset of images from a dataset and saves their corresponding image IDs to a CSV file.
        
        Params: dataset, a dataset of images consisting of image samples along with their corresponding label,
                subset, a subset of a dataset that contains a specific subset of indices from the original dataset. 
        Return: the subset is used to extract specific samples from "dataset"
        """
        fields = ["Image_ID"]
        rows = [[dataset.samples[i][0].split("/")[-1].replace(".jpg", "")] for i in subset.indices]

        with open("testing_ids3.csv", "w", newline = "") as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
    
    def prediction_to_csv(self, test_set: torch.utils.data.Subset, g: torch.Generator = None) -> None: 
        """
        Purpose: loads a trained cnn model, tests it on a given test set, and saves the predictions to a CSV file.
        Params: test_set, A subset of the original dataset containing unseen testing data,
                g, Sets the seed for the data loader workers for reproducibility.
        Returns: none
        """

        net = Net().to(self.device)

        print("Loading model from /model.pth!")

        net.load("./model.pth")
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 4, shuffle=False, num_workers = 2, worker_init_fn = self.seed_worker, generator = g)
        s_time = datetime.now()

        print("Testing model!")

        _, metrics = net.test_model(test_loader, self.device)
        self.predictions, self.test_truth = metrics[0], metrics[1]

        print(f"Finished testing in {datetime.now() - s_time}!")
        fields = ["CNN_Predictions"]
        
        rows = [[i] for i in self.predictions]

        with open("cnn_predictions.csv", "w", newline="") as f: 
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)

    def train_hold_out_split(self, train_set: torch.utils.data.Subset, validation_set: torchvision.datasets.ImageFolder, g: torch.Generator = None) -> None:
        """
        Purpose: Performs training and validation on a 80, 20 hold out split and saves the trained model.
        Params: train_set, a subset of the original dataset containing training data,
                validation_set, a dataset of images to be used for validation,
                g, sets the seed for the data loader workers for reproducibility.
        Returns: none.
        """

        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 6, worker_init_fn = self.seed_worker, generator = g)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 4, shuffle = False, num_workers = 2, worker_init_fn = self.seed_worker, generator = g)

        net = Net().to(self.device)

        s_time = datetime.now()

        self.training_loss, self.validation_loss = net.train_model(train_loader = train_loader, device = self.device, validation_loader = validation_loader)

        print(f"Done training in {datetime.now() - s_time} seconds!")

        net.save("model.pth")

        print("Model saved to /model.pth")

    def run_stratified_k_fold(self, dataset: torchvision.datasets.ImageFolder, seed: int = None, g: torch.Generator = None) -> None:
        """
        Purpose: performs stratified k-fold cross-validation on our dataset and calculates accuracy metrics for each fold. 
        Params: dataset, a dataset of images consisting of image samples along with their corresponding labels,
                seed,  sets the random seed for reproducibility when splitting the dataset into training and testing subsets,
                g, sets the seed for the data loader workers for reproducibility.
        Returns: none.
        """
        # define stratified k fold with 5 folds
        skf = StratifiedKFold(n_splits = 5, random_state = seed, shuffle = True)

        image_paths = [item[0] for item in dataset.samples]
        labels = [item[1] for item in dataset.samples]

        dataset_no_transformations, _ = self.download_data(testing = True)
        accuracies = []

        # iterate through five folds get predictions and accuracy
        for fold, (train_index, test_index) in enumerate(skf.split(image_paths, labels)):
            train_subset = torch.utils.data.Subset(dataset, train_index)
            test_subset = torch.utils.data.Subset(dataset_no_transformations, test_index)

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size = 4, shuffle = True, num_workers = 6, worker_init_fn = self.seed_worker, generator = g)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size = 4, shuffle = False, num_workers = 2, worker_init_fn = self.seed_worker, generator = g)

            net = Net().to(self.device)

            writer_1 = SummaryWriter(os.path.join("runs", f"isic_melanoma_train_fold-{fold}"))
            writer_2 = SummaryWriter(os.path.join("runs", f"isic_melanoma_metrics_fold-{fold}"))

            s_time = datetime.now()
            net.train_model(train_loader = train_loader, device = self.device)

            print(f"Done training in {datetime.now() - s_time} seconds!")

            accuracy, fold_metrics = net.test_model(test_loader = test_loader, device = self.device)

            print(f"Fold {fold + 1}: model achieved {accuracy}% accuracy on test set")
            
            accuracies.append(accuracy)
            self.predictions, self.test_truth = fold_metrics[0], fold_metrics[1]
            self.get_metrics(writer_2)
            
            print(f"Finished writing metrics for Fold {fold + 1}")
        # View metrics of stratified k fold with Tensorboard
        print(f"Run `tensorboard --logdir=./runs` to view metrics in-browser! Done testing.")

        np_array = np.array(accuracies)
        # Get mean accuracy and std
        print(f"Mean Accuracy: {np.mean(np_array)}")
        print(f"Sample stdev: {np.std(np_array, ddof = 1)}")

    @staticmethod
    def seed_worker(worker_id):
        """
        Purpose: ensures each worker has a unique, but reporducible random ssed. 
        Params: worker_id, unique identifier of the worker process.
        Returns: none.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def put_seed(self, seed):
        """
        Purpose: set a specific random seed for PyTorch Generator to get reproducible results
        Params: seed, an integer used to set the random seed
        Returns: g, PyTorch Generator object
        """
        print(f"Seed: {seed}")
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        g = torch.Generator()
        g.manual_seed(seed)

        return g

    def show_data(self, subset, g=None):
        """
        Purpose: Show a small batch of images with corresponding labels, from dataset used in PyTorch pipeline
        Params: subset, a Pytorch dataset to see images
                g, optional param used to control shuffling if seed is provided
        Returns: None
        """
        # get four images from pipeline dataset
        loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True, num_workers=6, worker_init_fn=self.seed_worker, generator=g)
        dataiter = iter(loader)
        images, labels = next(dataiter)
        image = torchvision.utils.make_grid(images)
        image = image / 2 + 0.5
        npimage = image.numpy()
        # save images to example.png
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        plt.savefig("example.png")

        print(" ".join(f"{self.classes[labels[j]]: 5d}" for j in range(4)))

        del(dataiter)

    def get_metrics(self, writer=None):
        """
        Purpose: model evaluation, to asses how well the model performs after training
                 obtains ROC curve, AUC curve, confusion matrix, precision/recall curve, and training/validation loss curves.
        Params: writer(optional), Tensorboard writer, used to log and visulaize metric
        Returns: None
        """
        fpr, tpr, _ = roc_curve(self.test_truth, self.predictions)

        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(self.test_truth, self.predictions)

        pr_auc = auc(recall, precision)

        c = confusion_matrix(self.test_truth, self.predictions)

        # confusion matrix setup
        plt.figure(figsize = (8, 6))
        sns.heatmap(c, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = [0, 1], yticklabels = [0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # save confusion matrix, ROC, annd PR data to tensorboard
        writer.add_figure("Confusion Matrix", plt.gcf(), global_step = 0)
        writer.add_hparams({"Metrics/ROC_AUC": roc_auc}, {})
        writer.add_hparams({"Metrics/PR_AUC": pr_auc}, {})

        # PR Curve
        np_test_truth = np.array(self.test_truth)
        no_skill = len(np_test_truth[np_test_truth == 1]) / len(np_test_truth)
        plt.figure()
        plt.plot(recall, precision, color = 'darkorange', lw = 2, label = 'PR curve (area = %0.2f)' % pr_auc)
        plt.plot([0, 1], [no_skill, no_skill], color = 'navy', lw = 2, linestyle = '--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Precision-Recall Curve (PR)")
        plt.legend(loc="lower right")
        # create PR curve visualization
        writer.add_figure("Precision Recall Curve", plt.gcf(), global_step = 0)

        # ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc = "lower right")
        # create ROC curve visualization
        writer.add_figure("Receiver Operating Characteristic Curve", plt.gcf(), global_step = 0)

        # Validation Loss Curve
        plt.figure()
        plt.plot(self.training_loss, label = "Training Loss")
        plt.plot(self.validation_loss, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        writer.add_figure("Validation & Training Loss Curve", plt.gcf(), global_step = 0)

def main():
    seed = 42

    # /home/xpeshla1/scratch/skin_lesion/train_v2/ contains 977 malignant and 977 benign examples = 1954 total
    # pipeline = Pipeline(log_path="./log.txt", download_path="/home/xpeshla1/scratch/skin_lesion/train_v2/", classes=(0, 1))
   
    # intialize pipeline
    pipeline = Pipeline(log_path="./log.txt", download_path="/scratch/xpeshla1/skin_lesion/train_og_sep_2020/", classes=(0, 1))

    pipeline.init_debug_logger()
    
    # set up randomization seed to get randomizaed but reproducible results
    if seed:
        g = pipeline.put_seed(seed)
    else:
        g = None

    # # training model
    # dataset, validation_set = pipeline.download_data(testing = False)

    # train_set, _ = pipeline.split_data(dataset = dataset, seed = seed)

    # pipeline.show_data(subset = train_set, g = g)

    # pipeline.train_hold_out_split(train_set = train_set, validation_set=validation_set, g = g)
    # # training model 

    # testing model with hold out
    # dataset, _ = pipeline.download_data(testing = True)

    # _, test_set = pipeline.split_data(dataset = dataset, seed = seed)

    # pipeline.split_subset_to_csv(dataset = dataset, subset = test_set)

    # pipeline.prediction_to_csv(test_set = test_set, g = g)

    # writer = SummaryWriter(os.path.join("runs", f"isic_melanoma_hold_out"))

    # pipeline.get_metrics(writer = writer)

    # testing model with stratified k-fold
    dataset, _ = pipeline.download_data(testing=False)
    pipeline.run_stratified_k_fold(dataset=dataset, seed=seed, g=g)

if __name__ == "__main__":
    main()