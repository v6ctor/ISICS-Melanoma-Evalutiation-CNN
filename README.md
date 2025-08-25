# Usage
Before executing any code from our project files, ensure the project is isolated in its own virtual environment. Follow the official Python documentation:
[Python Virtual Environment Guide](https://docs.python.org/3/library/venv.html)

If you're working on a Swarthmore CS Lab computer, you can refer to their specific guide:
[Swarthmore Virtual Environment Guide](https://www.cs.swarthmore.edu/newhelp/virtualenv.html)

Once the environment is set up, navigate to the /code directory and execute the following command in the terminal:
```python
pip install -r requirements.txt
```

## classifiers.ipynb
classifiers.ipynb contains all code for the five discrete classifiers that we used in this project. It contains functions to load the patient data, split into train and test sets based on what the CNN uses for its train and test sets, and rebalance the train set to contain balanced classes. After these initial functions, grid search hyperparameter tuning is conducted to find the best model for the patient data. The models are evaluated using confusion matrices, k-fold cross-validation, and Wilcoxon p-value tests. Lastly, the function to cross-compare the models' predictions with the CNN predictions is at the end. This section loads in the CNN predictions and showcases when the patient data model's predictions matched or did not match with the CNN predictions. 

To run this code, go to the end of the notebook and hit "Run All Above This Cell," then run the last cell. 

## classifier.py
The classifier.py file includes a PyTorch implementation of our convolutional neural network abstracted within a class. This class offers all essential functionalities for training, testing, validating, saving, and loading a pretrained model. To utilize this code as a **standalone** option, simply instantiate the Net class. pipeline.py automatically calls these methods at runtime in their own respective methods within the Pipeline class.

```python
from classifier import Net

nn = Net().to(torch.device("cuda"))

nn.load(...)
nn.train_model(...)
nn.test_model(...)
nn.save(...)
```

## pipeline.py
The pipeline.py file defines our pipeline for data preprocessing, training, evaluation, and testing phases of our CNN model. To promote good coding practices, we also decided to encapsulate our main functionality within a class, simplifying complexity and enhancing code readability.

### Training, validating, and testing a CNN Model
Due the modular nature of our code, there are many ways to run the training stage. The most simple is to use our `train_model` method that takes in a DataLoader PyTorch object for the training and (optionally) validation sets as well as the target device to send the computations to. Finally, use our `test_model` method to test the generalization of our model on unseen data.

Alternatively, you can use the following commands:
- `pipeline.py 42`: Fully retrain, validate, and test the model with a specified seed.
- `pipeline.py`: Retrain the model without using a seed.
- `pipeline.py 42 test`: Evaluate the model on the test set using a pre-saved model with a specified seed.
- `pipeline.py test`: Evaluate the model on the test set without seeding.

### Viewing Performance Metrics
We utilize TensorBoard to monitor our model's performance metrics during training and evaluation. To visualize metrics (confusion matrix, ROC curve, precision-recall curve, validation loss, and training loss), run the following command in a terminal within the image-classification directory.

```python
tensorboard --logdir=./runs
```
**WARNING**
We encountered issues with displaying results but resolved them by using an alternative command that forwards visuals through a different port.
```python
tensorboard --logdir=./runs --port 6002
```

