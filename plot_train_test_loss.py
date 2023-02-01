

import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load("logs/[2023-01-31-02-17-35].npz")
train_losses = npzfile['train_losses_per_epoch']
test_losses = npzfile['test_losses_per_epoch']

npzfile_bt = np.load("logs/[2023-01-31-01-45-20].npz")
train_losses_bt = npzfile_bt['train_losses_per_epoch']
test_losses_bt = npzfile_bt['test_losses_per_epoch']

plt.plot(range(len(train_losses)), train_losses, label="Train Imagenet")
plt.plot(range(len(test_losses)), test_losses, label="Test Imagenet")

plt.plot(range(len(train_losses_bt)), train_losses_bt, label="Train Barlow Twins")
plt.plot(range(len(test_losses_bt)), test_losses_bt, label="Test Barlow Twins")
plt.legend()
plt.show()
