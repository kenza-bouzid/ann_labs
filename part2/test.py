#%%
import dataset as ds
import mlp as mlp
import pandas as pd
import importlib
import matplotlib.pyplot as plt
#%%
importlib.reload(ds)
mg = ds.MackeyGlass()
df = mg.create_pandas_df()
# %%
df.head()
#%%
train, val, test = mg.split_train_val_test_tf()
#%%
train
# %%
x = mg.generate_x()
plt.plot(range(301, 1500),mg.x[301:1500])
plt.xlabel('Time')
plt.ylabel('Time series')
#%%
importlib.reload(mlp)
mlp1 = mlp.MLP()
model = mlp1.set_model()
mlp1.compile()
#%%
history = mlp1.train(train, val, epochs=10)

#%%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#%%
mlp1.plot_loss()
#%%
model.summary()
#%%
model.evaluate(test)
#%%
y_pred = model.predict(test)
plt.plot(range(200), df["x+5"][1000:])
# plt.plot(range(200), y_pred)
plt.xlabel('Time')
plt.ylabel('Time series')
