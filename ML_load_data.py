import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("/mnt/d/phD/CS materials/Advanced ML/kenya_kilifi_w1_clean_08.03.2026.csv")
cols_to_remove = ["phq9_total_score",	"psq_total_score",	"Anxiety Severity",	"Anxiety Yes/No","Depression Severity"]
X = dataset.drop(columns=cols_to_remove).values
y = dataset["Depression Yes/No"].values