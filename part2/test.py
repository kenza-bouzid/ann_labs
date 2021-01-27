#%%
import dataset as ds
import mlp as mlp
import hparam_tuning as ht
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import tensorflow as tf
#%%
importlib.reload(ds)
mg = ds.MackeyGlass()
df = mg.create_pandas_df()
# %%
## Checking gpu setup
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))
# %%
df.head()
#%%
X, y = mg.get_data()
X_train, X_val, X_test = X[:900], X[900:1000], X[1000:1200]
y_train, y_val, y_test = y[:900], y[900:1000], y[1000:1200]

# %%
x = mg.generate_x()
plt.plot(range(301, 1500),mg.x[301:1500])
plt.xlabel('Time')
plt.ylabel('Time series')
#%%
importlib.reload(mlp)
mlp1 = mlp.MLP(X_train, y_train, X_val, y_val, X_test, y_test)
model = mlp1.set_model()
mlp1.compile()

#%%

#%%
with tf.device('/device:GPU:0'):
    history = mlp1.train(epochs=10)

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
plt.legend()
#%%
# HYPERPARAM TUNING
importlib.reload(ht)
tuning = ht.HparamTuning(X_train, y_train, X_val, y_val, X_test, y_test)
#%%
with tf.device('/device:GPU:0'):
  tuning.run_hparam_tuning()