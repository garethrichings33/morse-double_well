def convert_data_to_numpy(data_frame):
    '''
    Extract data from a pandas DataFrame to two numpy arrays:
    X is an array of the coordinates.
    Y is an array of the corresponding function values.
    '''
    X = data_frame.iloc[:, 0:2].to_numpy()
    Y = data_frame['value'].to_numpy()

    return X, Y


def create_dataset(X, Y):
    '''
    Take two arrays containing coordinates (X) and corresponding 
    function values (Y) and combine them into a DataSet object.
    '''
    from numpy import float32
    from torch import tensor
    from torch.utils.data import TensorDataset

    X_t = tensor(X.astype(float32))
    Y_t = tensor(Y.astype(float32))

    return TensorDataset(X_t, Y_t)


def plot_losses(training_loss_tracker, validation_loss_tracker):
    '''
    Plot progress of training and validation losses against epoch number.
    '''
    from matplotlib import pyplot as plt

    plt.figure()
    epochs = []
    loss_values = []
    for i in range(len(training_loss_tracker)):
        epoch, loss = training_loss_tracker[i]
        epochs.append(epoch)
        loss_values.append(loss)
    ax = plt.axes()
    ax.scatter(epochs, loss_values, marker='o', label='Training')

    epochs = []
    vloss_values = []
    for i in range(len(validation_loss_tracker)):
        epoch, vloss = validation_loss_tracker[i]
        epochs.append(epoch)
        vloss_values.append(vloss)
    ax.scatter(epochs, vloss_values, marker='x', label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend(loc='upper right')
    plt.show()


def extract_surface_data_to_lists(dataset, model):
    '''
    Take a Dataset and NN model and extract the coordinates, 
    and actual and predicted function values to lists.
    '''
    from torch import no_grad

    x1_list = []
    x2_list = []
    fn_value_list = []
    prediction_list = []
    with no_grad():
        for i in range(len(dataset)):
            coordinates, fn_value = dataset[i]
            x1, x2 = coordinates.numpy()[0], coordinates.numpy()[1]
            x1_list.append(x1)
            x2_list.append(x2)
            fn_value_list.append(fn_value.numpy())
            prediction_list.append(model(coordinates).numpy()[0])

    return x1_list, x2_list, fn_value_list, prediction_list


def plot_fit_vs_values(dataset, model):
    '''
    Plot data against predicted responses.
    '''
    from matplotlib import pyplot as plt

    x1, x2, fn_value_list, prediction_list = extract_surface_data_to_lists(
        dataset, model)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x1, x2, fn_value_list,
                 c=fn_value_list, cmap='Greens')
    ax.scatter3D(x1, x2, prediction_list,
                 c=prediction_list, cmap='Reds')
    plt.show()


def train_one_epoch(model, training_loader, optimiser, loss_fn, len_dataset):
    '''
    Function to train a single epoch.
    '''
    running_loss = 0.

    for data in training_loader:
        inputs, fn_values = data
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), fn_values)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    return running_loss/len_dataset


def get_validation_loss(model, validation_loader, loss_fn, len_dataset):
    import torch

    running_vloss = 0.
    with torch.no_grad():
        for vdata in validation_loader:
            v_inputs, v_fn_values = vdata
            v_outputs = model(v_inputs)
            vloss = loss_fn(v_outputs.squeeze(), v_fn_values)
            running_vloss += vloss.item()
        validation_loss = running_vloss/(len_dataset)
    return validation_loss


def fit_network(filename):
    import math
    import pandas as pd
    from sklearn.model_selection import train_test_split

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from pytorch_lightning import seed_everything

# Random seed to ensure repeatability when testing.
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

# Get data from CSV file.
    data_frame = pd.read_csv(filename)
# Extract coordinates and function values to numpy arrays.
    coordinates, responses = convert_data_to_numpy(data_frame)

# Split data into training and validation sets, fixing the random state.
    (coordinates_train,
     coordinates_valid,
     responses_train,
     responses_valid) = train_test_split(coordinates,
                                         responses,
                                         test_size=0.2,
                                         # Fix split for repeatability when testing.
                                         random_state=10)

# Setup the training and validation DataSets.
    surface_train_dataset = create_dataset(coordinates_train, responses_train)
    surface_valid_dataset = create_dataset(coordinates_valid, responses_valid)

# Define the training and validation DataLoaders.
    batch_size = 20
    training_loader = DataLoader(surface_train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0)

    validation_loader = DataLoader(surface_valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=0)

# Define the neural network.
    class SurfaceModel(nn.Module):
        def __init__(self):
            super(SurfaceModel, self).__init__()
            self.activation = nn.Tanh()
            self.linear1 = nn.Linear(2, 20)
            self.linear2 = nn.Linear(20, 20)
            self.linear_out = nn.Linear(20, 1)

        def forward(self, x):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            x = self.linear_out(x)
            return x

# Create the objects needed for training.
    model = SurfaceModel()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(
        model.parameters(), lr=0.0002, weight_decay=0., momentum=0.4)

# Training
# Loops to train multiple epochs.
    EPOCHS = 10_000
    training_loss_tracker = []
    validation_loss_tracker = []
    validation_loss_min = 1_000_000
    for epoch in range(EPOCHS):
        model.train(True)
        training_loss = train_one_epoch(model,
                                        training_loader,
                                        optimiser,
                                        loss_fn,
                                        len(surface_train_dataset))

        model.eval()
        validation_loss = get_validation_loss(model, validation_loader,
                                              loss_fn, len(surface_valid_dataset))
        if validation_loss < validation_loss_min:
            validation_loss_min = validation_loss
            validation_loss_min_epoch = epoch

        print(f'Epoch: {epoch+1}, Training Loss: {training_loss:14.12f}, '
              f'Validation Loss: {validation_loss:14.12f}')
        if ((epoch+1) % 50 == 0):
            training_loss_tracker.append(
                (epoch+1, math.log(training_loss, 10)))
            validation_loss_tracker.append(
                (epoch+1, math.log(validation_loss, 10)))

    print(f'Minimum validation loss: {validation_loss_min:14.12f} '
          f'at epoch {validation_loss_min_epoch}')

# Plot progress of training and validation losses
    plot_losses(training_loss_tracker, validation_loss_tracker)

# Plot training data vs predictions
    plot_fit_vs_values(surface_train_dataset, model)
# Plot validation data vs predictions
    plot_fit_vs_values(surface_valid_dataset, model)


if __name__ == '__main__':
    from generate_data import generate_data

    filename = 'surface.csv'
    generate_data(filename)
    fit_network(filename)
