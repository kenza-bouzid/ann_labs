import tensorflow as tf
import matplotlib.pyplot as plt

tfk = tf.keras
tfkl = tfk.layers


class MLP():
    def __init__(self, n_input=5, n_output=1, nh1=3, nh2=2):
        self.n_input = n_input
        self.n_output = n_output
        self.nh1 = nh1
        self.nh2 = nh2

    def set_model(self):
        self.model = tf.keras.models.Sequential([
            tfkl.Dense(self.n_input, activation='sigmoid',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tfkl.Dense(self.nh1, activation='sigmoid',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tfkl.Dense(self.nh2, activation='sigmoid',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tfkl.Dense(units=self.n_output)
        ])
        return self.model

    def compile(self, lr=1e-3, momentum=0.9):
        optimizer = tfk.optimizers.SGD(lr=lr, momentum=momentum)
        self.model.compile(loss="mse",
                           optimizer=optimizer, metrics=["mse"])

    def train(self, train_set, validation_set, batch_size=50, epochs=100):
        es_callback = tfk.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001,
                                                  patience=20,
                                                  verbose=1,
                                                  restore_best_weights=True)
        self.history = self.model.fit(
            train_set, epochs=epochs, verbose=1, 
            callbacks=[es_callback], validation_data=validation_set)
        return self.history

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)