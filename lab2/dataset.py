import numpy as np

def generate_x(start, end, start_test, step):
    x = []
    x_test = []
    cur_val = start
    while cur_val <= end:
      x.append(cur_val)
      x_test.append(cur_val+start_test)
      cur_val += step

    return x, x_test



class SinusData():
  def __init__(self, start=0, end=2*np.pi, start_test=0.05, step=0.1, noise=False) -> None:
    x, x_test = generate_x(start, end, start_test, step)

    self.x = 2*np.array(x)
    self.x_test = 2*np.array(x_test)
    self.y = np.sin(self.x)
    self.y_test = np.sin(self.x_test)
    
    if noise:
      noise = np.random.normal(0, 0.1 , len(x))
      self.x += noise
      noise_test = np.random.normal(0, 0.1 , len(x))
      self.x_test += noise_test

class SquareData():
  def __init__(self, start=0, end=2*np.pi, start_test=0.05, step=0.1, noise=False) -> None:
    x, x_test = generate_x(start, end, start_test, step)

    self.x = 2*np.array(x)
    self.x_test = 2*np.array(x_test)
    self.y = np.array([0 if sin_val < 0 else 1 for sin_val in np.sin(self.x)])
    self.y_test = np.array([0 if sin_val < 0 else 1 for sin_val in np.sin(self.x_test)])

    if noise:
      noise = np.random.normal(0, 0.1 , len(x))
      self.x += noise
      noise_test = np.random.normal(0, 0.1 , len(x))
      self.x_test += noise_test
