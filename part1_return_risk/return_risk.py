import pandas as pd

df = pd.read_csv("orders_dataset.csv")

print("Dataset shape:", df.shape)
print("\nReturn rate:", df["returned"].mean())