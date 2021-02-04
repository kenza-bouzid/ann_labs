#%%
import dataset as ds
import importlib
importlib.reload(ds)
mg = ds.MackeyGlass()
train, val, test = mg.get_train_val_test()
# %%
for features_tensor, target_tensor in test:
    print(f'features:{features_tensor} target:{target_tensor}')
# %%
mg.df.head(5)