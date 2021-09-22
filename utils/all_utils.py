import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")



def prepare_data(df):
  x=df.drop("y",axis=1)

  y=df["y"]

  return x,y

def save_model(model,filename):
  model_dir="models"
  os.makedirs(model_dir,exist_ok=True)
  filePath=os.path.join(model_dir,filename)
  joblib.dump(model,filePath)