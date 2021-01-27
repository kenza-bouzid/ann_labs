
def class_accuracy(y, y_pred, class_label=1):
  class_samples_num = (y == class_label).sum()
  correclty_predicted_samples_num = ((y_pred == class_label) & (y_pred == y)).sum()
  return correclty_predicted_samples_num / class_samples_num

def accuracy(y, y_pred):
  accA = class_accuracy(y, y_pred, -1)
  accB = class_accuracy(y, y_pred, 1)
  print(f'Accuracy for A: {accA}, Accuracy for B: {accB}')