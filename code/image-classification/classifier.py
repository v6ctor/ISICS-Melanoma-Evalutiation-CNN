import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        """
        Defines a neural network architecture with convolutional and fully connected
        layers for image classification.
        """
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.2)

        # (input channels, out channels, sizee of kernel)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # size of window to perform pooling, stride (how much window moves during pooling)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        """
        Defines a forward pass for a neural network model with convolutional and fully
        connected layers, using ReLU activation functions and dropout regularization.
        
        :param x: the input data that will be passed through the layers of the
        model during the forward pass. Each line in the `forward` method corresponds to a layer or operation
        applied
        :return: the output of the neural network model after passing the
        input `x` through several layers including convolutional layers, activation functions (ReLU),
        pooling layers, dropout layers, and fully connected layers.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def train_model(self, train_loader, device, validation_loader=None):
        """
        Trains our neural network model, logging the training loss at regular intervals.
        
        :param train_loader: a DataLoader object that provides batches of training data to the model during the 
        training process. 
        :param device: a string "cpu" or "cuda" that specifies whether the model should be trained on a CPU or a GPU.
        :param writer_train: a TensorBoard SummaryWriter object logs training metrics such as loss, accuracy, etc. 
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        training_loss = []
        validation_loss = []

        # set model to "train" mode for training
        self.train()
        for epoch in range(0, 10):
            running_loss = 0.0
            for _, data in enumerate(train_loader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # calculate metrics
                running_loss += loss.item() * images.size(0)

            running_loss /= len(train_loader.dataset)
            training_loss.append(running_loss)

            print(f"[Epoch {epoch + 1}/10] Training Loss: {running_loss:.6f}%")

            if validation_loader is not None:
                val_loss_epoch = self.validate_model(validation_loader=validation_loader, device=device, epoch=epoch)
                validation_loss.append(val_loss_epoch)
                
        return training_loss, validation_loss

    def validate_model(self, validation_loader, device, epoch):
        """
        Validates the model on the validation dataset, computing and logging accuracy and loss.
        
        :param validation_loader: a DataLoader object that provides batches of validation data to the model during the 
        training process at every epoch. 
        :param device: a string "cpu" or "cuda" that specifies whether the model should be trained on a CPU or a GPU.
        :param writer_validation: a TensorBoard SummaryWriter object that logs validation metrics.
        :param epoch: the current epoch number of a model during the training process.
        """
        correct = 0
        total = 0
        running_loss = 0.0

        criterion = nn.CrossEntropyLoss()
        
        # set model to "eval" mode for evaluating our model's fit
        self.eval()

        with torch.no_grad():
            for data in validation_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                
                # update loss
                running_loss += loss.item() * images.size(0)
                
                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # calculate val loss and accuracy
        running_loss /= len(validation_loader.dataset)
        mean_validation_accuracy = 100 * (correct / total)

        print(f"[Epoch {epoch + 1}/10] Validation Loss: {running_loss:.6f}, Accuracy: {mean_validation_accuracy:.6f}%")

        return running_loss

    def test_model(self, test_loader, device):
        """
        Evaluates our neural network model's performance on a test dataset and returns the accuracy
        and predictions.
        
        :param test_loader:  a DataLoader object that provides batches of test data to evaluate the model. 
        It contains the test images and their corresponding labels
        :param device: a string "cpu" or "cuda" that specifies whether the model should be trained on a CPU or a GPU.
        :return: the accuracy of the network on the test images as a percentage, 
        and a list containing two elements: a list of predictions and a list of ground truth
        labels for the test images in order that they were tested in.
        """
        predictions = []
        test_truths = []

        correct = 0
        total = 0

        # set model to "eval" mode for testing
        self.eval()

        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predictions.extend(predicted.tolist())
                test_truths.extend(labels.tolist())
                
        print(f'Accuracy of the network on the test images: {(100 * correct  / total):.6f} %')

        return correct / total, [predictions, test_truths]

    def save(self, path):
        """
        Saves the state dictionary of our neural network using a specified path.
        
        :param path: the file path where the model's state dictionary will be saved
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads a saved state dictionary from a specified path.
        
        :param path: the file path from which the model state dictionary will be loaded
        """
        self.load_state_dict(torch.load(path))