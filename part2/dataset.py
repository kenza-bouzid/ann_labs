
import tensorflow as tf
import pandas as pd
import numpy as np

class MackeyGlass():
    def __init__(self, x0=1.5, n_samples=1200, start=301, end=1500):
        self.x0 = x0
        self.n_samples = n_samples
        self.start = start
        self.end = end
        self.x = None
        self.df = None

    def generate_x(self, noise=False, sigma=0.05):    
        self.x = [self.x0]
        for t in range(1, self.end + 6):
            x25 = self.x[t-25] if t >= 25 else 0
            self.x.append(self.x[t-1] + 0.2*x25/(1+x25**10) - 0.1*self.x[t-1])
        if noise:
            noise = np.random.normal(0, sigma , self.end+6)
            self.x = self.x + noise
        return self.x 

    def create_pandas_df(self):

        if self.df:
            return self.df

        x = self.generate_x()
        data = {
            "x-20": [],
            "x-15": [],
            "x-10": [],
            "x-5": [],
            "x-0": [],
            "x+5": []
        }
        for t in range(self.start, self.end+1):
            for i in range(0, 21, 5):
                data[f'x-{i}'].append(x[t-i])

            data["x+5"].append(x[t+5])

        self.df = pd.DataFrame(data=data)
        self.df.to_csv("data/mackey_glass.csv")
        return self.df

    def get_data(self, noise=False, sigma=0.05):
        x = self.generate_x(noise, sigma)
        features = []
        labels = []
        
        for t in range(self.start, self.end+1):
            features.append(
                np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]],  dtype="float32"))
            labels.append(x[t+5])

        features = np.array(features, dtype="float32")
        labels = np.array(labels, dtype="float32")

        return features, labels

    def get_df(self, file_name="data/mackey_glass.csv"):
        self.df = pd.read_csv(file_name)
        return self.df

    def split_train_val_test(self, df):
        self.train_df, self.validation_df, self.test_df = df[:900], df[900:1000], df[1000:]
        return self.train_df, self.validation_df, self.test_df

    def split_train_val_test_tf(self):
        train, validation, test = self.split_train_val_test(self.get_df())
        features = ["x-20", "x-15", "x-10", "x-5", "x-0"]
        self.train_set = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(train[features].values, tf.float32),
                    tf.cast(train['x+5'].values, tf.float32)
                )
            )
        )
        self.validation_set = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(validation[features].values, tf.float32),
                    tf.cast(validation['x+5'].values, tf.float32)
                )
            )
        )
        self.test_set = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(test[features].values, tf.float32),
                    tf.cast(test['x+5'].values, tf.float32)
                )
            )
        )
        return self.train_set, self.validation_set, self.test_set


def main():
    mg = MackeyGlass()
    train, _, _ = mg.get_train_val_test()
    for features_tensor, target_tensor in train:
        print(f'features:{features_tensor} target:{target_tensor}')

if __name__ == "__main__":
    main()
