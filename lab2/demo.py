from dataset import SinusData
from dataset import SquareData
import matplotlib.pyplot as plt

sin_data = SinusData(noise=True)
plt.scatter(sin_data.x, sin_data.y)
plt.show()

sqr_data = SquareData(noise=True)
plt.scatter(sqr_data.x, sqr_data.y)
plt.show()