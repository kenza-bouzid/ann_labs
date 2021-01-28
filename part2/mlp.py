from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
tfk = tf.keras
tfkl = tfk.layers

class MLP():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, n_input=5, n_output=1, nh1=5, nh2=6, lambda_=0.01):
        self.n_input = n_input
        self.n_output = n_output
        self.nh1 = nh1
        self.nh2 = nh2
        self.lambda_ = lambda_
        self.X_train = X_train
        self.y_train = y_train
        self.X_val= X_val 
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def set_model(self):
        self.model = tf.keras.models.Sequential([
            tfkl.Dense(self.n_input, activation='elu', name="input_layer",
                       kernel_regularizer=tfk.regularizers.l2(self.lambda_),
                       kernel_initializer="he_normal"),
            tfkl.Dense(self.nh1, activation='elu', name="hidden_layer1",
                       kernel_regularizer=tfk.regularizers.l2(self.lambda_),
                       kernel_initializer="he_normal"),
            tfkl.Dense(self.nh2, activation='elu', name="hidden_layer2",
                       kernel_regularizer=tfk.regularizers.l2(self.lambda_),
                       kernel_initializer="he_normal"),
            tfkl.Dense(units=self.n_output, name="output_layer",
                       kernel_regularizer=tfk.regularizers.l2(self.lambda_),
                       kernel_initializer="glorot_normal")
        ])
        return self.model

    def compile(self, lr=1e-2, momentum=0.9):
        optimizer = tfk.optimizers.SGD(lr=lr, momentum=momentum)
        self.model.compile(loss="mse",
                           optimizer=optimizer, metrics=["mse"])

    def train(self, batch_size=50, epochs=100):
        es_callback = tfk.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001,
                                                  patience=20,
                                                  verbose=1,
                                                  restore_best_weights=True)
        
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tfk.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        
        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      batch_size=batch_size, 
                                      epochs=epochs, verbose=1,
                                      callbacks=[es_callback, tensorboard_callback], 
                                      validation_data=(self.X_val, self.y_val))
        return self.history

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
