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

class XData():
    def __init__(self, start=0, end=2*np.pi, start_test=0.05, step=0.1, noise=False, seed=42) -> None:
        np.random.seed(seed)
        x, x_test = generate_x(start, end, start_test, step)

        self.x = np.array(x)
        self.x_test = np.array(x_test)

        if noise:
            noise = np.random.normal(0, 0.1, len(x))
            self.x += noise
            noise_test = np.random.normal(0, 0.1, len(x))
            self.x_test += noise_test


class SinusData(XData):
    def __init__(self, start=0, end=2*np.pi, start_test=0.05, step=0.1, noise=False, seed=42) -> None:
        super().__init__(start, end, start_test, step, noise, seed)
        self.y = np.sin(2*self.x)
        self.y_test = np.sin(2*self.x_test)

class SquareData(XData):
    def __init__(self, start=0, end=2*np.pi, start_test=0.05, step=0.1, noise=False, seed=42) -> None:
        super().__init__(start, end, start_test, step, noise, seed)
        self.y = np.array([0 if sin_val < 0 else 1 for sin_val in np.sin(2*self.x)])
        self.y_test = np.array([0 if sin_val < 0 else 1 for sin_val in np.sin(2*self.x_test)])
