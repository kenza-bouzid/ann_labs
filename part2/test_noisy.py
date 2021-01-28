#%%
import dataset as ds
import mlp as mlp
import hparam_tuning_noise as hts
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import tensorflow as tf
# %%
## Checking gpu setup
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))
#%%
importlib.reload(ds)
mg = ds.MackeyGlass()
# %%
x = mg.generate_x(True, 0.05)
plt.plot(range(301, 1500), mg.x[301:1500])
plt.xlabel('Time')
plt.ylabel('Time series')
plt.title('Noisy Mackey Glass Time Series data for t in [301:1500], sigma=0.05')
plt.savefig("images/time_series_noise_05.png")
# %%
x = mg.generate_x(True, 0.15)
plt.plot(range(301, 1500), mg.x[301:1500])
plt.xlabel('Time')
plt.ylabel('Time series')
plt.title(
    'Noisy Mackey Glass Time Series data for t in [301:1500], sigma=0.15')
plt.savefig("images/time_series_noise_15.png")
#%%
X_noisy, y_noisy = mg.get_data(True, 0.15)
X, y = mg.get_data()

X_train, X_val, X_test = X_noisy[:900], X[900:1000], X[1000:1200]
y_train, y_val, y_test = y_noisy[:900], y[900:1000], y[1000:1200]

#%%
importlib.reload(mlp)
mlp1 = mlp.MLP(X_train, y_train, X_val, y_val, X_test, y_test, nh1=4, nh2=2, lambda_=0.0001)
model = mlp1.set_model()
mlp1.compile(lr=0.05, momentum=0.9)
#%%
with tf.device('/device:GPU:0'):
    history = mlp1.train(epochs=500)

#%%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#%%
mlp1.plot_loss()
#%%
model.summary()
#%%
model.evaluate(X_test, y_test)
#%%
y_pred = model.predict(X_test)
plt.plot(range(200), y_test, label="Known Targets")
plt.plot(range(200), y_pred, label="Predicted Targets")
plt.xlabel('Time')
plt.ylabel('Time series')
plt.title("Test predictions along with the known target values with noisy train data.")
plt.legend()
plt.savefig("images/noisy15.png")
#%%
# HYPERPARAM TUNING
importlib.reload(hts)
tuning = hts.HparamTuning('logs\\hparam_tuning_noisy_15\\',
                         X_train, y_train, X_val, y_val, X_test, y_test)
#%%
with tf.device('/device:GPU:0'):
  tuning.run_hparam_tuning()
