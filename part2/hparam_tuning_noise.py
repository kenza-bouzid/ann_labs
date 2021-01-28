from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

HP_NH2 = hp.HParam('nh2', hp.Discrete([2, 4, 6]))
HP_LAMBDA = hp.HParam('lambda', hp.Discrete([0.001, 0.005, 0.01, 0.05, 0.1]))

NH1 = 4
ALPHA = 0.8
LR = 0.05
METRIC_MSE = 'mse'
N_INPUT = 5
N_OUTPUT = 1
BATCH_SIZE = 50
EPOCHS = 500
LOG_DIR = 'logs\\hparam_tuning_noisy\\'
class HparamTuning():
    def __init__(self, log_dir, X_train, y_train, X_val, y_val, X_test, y_test):
        self.log_dir = log_dir
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


    def hyperparameter_setup(self):
        tf.summary.trace_on(graph=True, profiler=True)
        with tf.summary.create_file_writer(self.log_dir).as_default():
            hp.hparams_config(
                hparams=[HP_NH2, HP_LAMBDA],
                metrics=[hp.Metric(METRIC_MSE, display_name='mse')],
            )

    def train_test_model(self, log_dir, hparams):
        model = tf.keras.models.Sequential([
            tfkl.Dense(N_INPUT, activation='elu', name="input_layer",
                       kernel_regularizer=tfk.regularizers.l2(hparams[HP_LAMBDA]),
                       kernel_initializer="he_normal"),
            tfkl.Dense(NH1, activation='elu', name="hidden_layer1",
                       kernel_regularizer=tfk.regularizers.l2(hparams[HP_LAMBDA]),
                       kernel_initializer="he_normal"),
            tfkl.Dense(hparams[HP_NH2], activation='elu', name="hidden_layer2",
                       kernel_regularizer=tfk.regularizers.l2(hparams[HP_LAMBDA]),
                       kernel_initializer="he_normal"),
            tfkl.Dense(units=N_OUTPUT, name="output_layer",
                       kernel_regularizer=tfk.regularizers.l2(hparams[HP_LAMBDA]),
                       kernel_initializer="glorot_normal")
        ])

        optimizer = tfk.optimizers.SGD(
            lr=LR, momentum=ALPHA)
        model.compile(loss="mse",
                           optimizer=optimizer, metrics=["mse"])
        
        tensorboard_callback = tfk.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        # hp_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

        history = model.fit(x=self.X_train, y=self.y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS, verbose=1,
                                callbacks=[tensorboard_callback],
                                validation_data=(self.X_val, self.y_val))
        
        _, mse = model.evaluate(self.X_test, self.y_test)
        return mse

    def run(self, run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            mse = self.train_test_model(run_dir, hparams)
            tf.summary.scalar(METRIC_MSE, mse, step=1)

    def run_hparam_tuning(self):
        with tf.device('/device:GPU:0'):
            self.hyperparameter_setup()
            session_num = 0
            for nh2 in HP_NH2.domain.values:
                for lambda_ in HP_LAMBDA.domain.values:
                    hparams = {
                        HP_NH2: nh2,
                        HP_LAMBDA: lambda_,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    self.run(
                        self.log_dir + run_name, hparams)
                    session_num += 1
