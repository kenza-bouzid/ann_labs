from rbf import LearningMode
from rbf import CentersSampling
from dataset import SinusData
from dataset import SquareData
from utils_test import *


## SINUS EXPERIMENTS ##
sin_data = SinusData(noise=False)
# error_experiment(sin_data, "sinus", LearningMode.BATCH, CentersSampling.WEIGHTED)
# plot_estimate(sin_data, type="sinus", learning_mode=LearningMode.BATCH,
#               centers_sampling=CentersSampling.LINEAR, n=20, sigma=1.0)
plot_estimate(sin_data, type="sinus", learning_mode=LearningMode.BATCH,
              centers_sampling=CentersSampling.WEIGHTED,n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0)

## SQUARE EXPERIMENTS ##
sqr_data = SquareData(noise=False)
# error_experiment(sqr_data, "square", LearningMode.BATCH,
#                  CentersSampling.WEIGHTED)
# plot_estimate(sqr_data, type="square", learning_mode=LearningMode.BATCH,
#               centers_sampling=CentersSampling.LINEAR, n=20, sigma=1.0)
plot_estimate(sqr_data, type="square", learning_mode=LearningMode.BATCH,
              centers_sampling=CentersSampling.WEIGHTED, n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0)
