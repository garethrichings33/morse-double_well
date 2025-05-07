if __name__ == '__main__':
    import math
    import pandas as pd
    import numpy as np
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt

    from sklearn.model_selection import train_test_split

    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from pytorch_lightning import seed_everything
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

# Get the data from file
    data_frame = pd.read_csv('surface.csv')
    X = data_frame.iloc[:, 0:2].to_numpy()
    Y = data_frame['value'].to_numpy()


# Split data into training and validation sets, fixing the random state.
    (X_train,
     X_valid,
     Y_train,
     Y_valid) = train_test_split(X,
                                 Y,
                                 test_size=0.2,
                                 random_state=10)

# Setup the training and validation datasets.
    X_train_t = torch.tensor(X_train.astype(np.float32))
    Y_train_t = torch.tensor(Y_train.astype(np.float32))
    surface_train_dataset = TensorDataset(X_train_t, Y_train_t)

    X_valid_t = torch.tensor(X_valid.astype(np.float32))
    Y_valid_t = torch.tensor(Y_valid.astype(np.float32))
    surface_valid_dataset = TensorDataset(X_valid_t, Y_valid_t)

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
            self.linear1 = nn.Linear(2, 20)
            self.activation = nn.Tanh()
            self.linear2 = nn.Linear(20, 20)
            # self.linear3 = nn.Linear(30, 10)
            self.linear_out = nn.Linear(20, 1)

        def forward(self, x):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            # x = self.activation(self.linear3(x))
            x = self.linear_out(x)
            return x

    model = SurfaceModel()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(
        model.parameters(), lr=0.0002, weight_decay=0., momentum=0.4)

# Training
    def train_one_epoch():
        running_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, fn_values = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), fn_values)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        return running_loss/len(surface_train_dataset)

    EPOCHS = 30_000
    training_loss_tracker = []
    validation_loss_tracker = []
    validation_loss_min = 1_000_000
    for epoch in range(EPOCHS):
        model.train(True)
        training_loss = train_one_epoch()

        model.eval()
        running_vloss = 0.

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                v_inputs, v_fn_values = vdata
                v_outputs = model(v_inputs)
                vloss = loss_fn(v_outputs.squeeze(), v_fn_values)
                running_vloss += vloss.item()
            validation_loss = running_vloss/(len(surface_valid_dataset))
            if validation_loss < validation_loss_min:
                validation_loss_min = validation_loss
                validation_loss_min_epoch = epoch

        print(f'Epoch: {epoch+1}, Training Loss: {training_loss},\
              Validation Loss: {validation_loss}')
        if ((epoch+1) % 50 == 0):
            training_loss_tracker.append(
                (epoch+1, math.log(training_loss, 10)))
            validation_loss_tracker.append(
                (epoch+1, math.log(validation_loss, 10)))

    print(f'Minimum validation loss: {validation_loss_min} \
        at epoch {validation_loss_min_epoch}')

# Plot progress of training and validation losses
    fig = plt.figure()
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

    exit()

# Plot training data against predictions.
    xdata = []
    ydata = []
    training_data_plot = []
    prediction_data_plot = []
    with torch.no_grad():
        for i in range(len(dataset)):
            coordinates, training_value = dataset[i]
            x, y = tuple(coordinates.numpy())
            xdata.append(x)
            ydata.append(y)
            training_data_plot.append(training_value.numpy()[0])
            prediction_data_plot.append(model(coordinates).numpy()[0])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, training_data_plot,
                 c=training_data_plot, cmap='Greens')
    ax.scatter3D(xdata, ydata, prediction_data_plot,
                 c=prediction_data_plot, cmap='Reds')
    plt.show()
