import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

data=pd.read_csv("C:/Python/datasets/train.csv/train.csv")
print(data.head())
data=np.array(data)