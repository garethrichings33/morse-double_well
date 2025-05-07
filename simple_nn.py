if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt

    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from pytorch_lightning import seed_everything
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

# Get the DataSet

    class SurfaceDataset(Dataset):
        def __init__(self, csv_file):
            super().__init__()
    # Read CSV file of data.
            self.data_frame = pd.read_csv('surface.csv')

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, idx):
            data = self.data_frame.iloc[idx]
            coordinates = torch.from_numpy(
                np.array([data['x'], data['y']])).to(torch.float32)
            value = torch.tensor(data['value']).to(torch.float32).unsqueeze(0)
            return coordinates, value

    dataset = SurfaceDataset('surface.csv')

# Define the DataLoader
    batch_size = 10
    training_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0)

    class SurfaceModel(nn.Module):
        def __init__(self):
            super(SurfaceModel, self).__init__()
            self.linear1 = nn.Linear(2, 20)
            self.activation = nn.Sigmoid()
            self.linear2 = nn.Linear(20, 1)
            # self.linear3 = nn.Linear(10, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            # x = self.activation(x)
            # x = self.linear3(x)
            return x

    model = SurfaceModel()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)

# Fitting
    def train_one_epoch(epoch_index):
        running_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, answer = data
            optimiser.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, answer)
            loss.backward()
            optimiser.step()
            running_loss += loss

        return running_loss/len(dataset)

    EPOCHS = 500
    loss_tracker = []
    for epoch in range(EPOCHS):
        model.train()
        avg_loss = train_one_epoch(epoch+1)
        model.eval()
        print(f'Epoch: {epoch+1}, Loss: {avg_loss}')
        if (epoch % 50 == 0):
            loss_tracker.append((epoch, avg_loss.item()))

    model.eval()

# Plot progress of training loss
    fig = plt.figure()
    epochs = []
    loss_values = []
    for i in range(len(loss_tracker)):
        epoch, loss = loss_tracker[i]
        epochs.append(epoch)
        loss_values.append(loss)
    ax = plt.axes()
    ax.scatter(epochs, loss_values)
    plt.show()


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
