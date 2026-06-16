import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path
Path("data").mkdir(exist_ok=True)
iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv("data/iris.csv", index=False)
print("Dataset saved to data/iris.csv")
