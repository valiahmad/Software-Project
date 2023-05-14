import pandas as pd
import numpy as np


def concat(df: pd.DataFrame, col: str, i: int):
    if (len(df)-1) == i:
        return df[col].iloc[i]
    return np.concatenate((df[col].iloc[i], concat(df, col, i+1)), axis=0)


arr = np.array([
    [1,2],
    [3,4],
    [5,6],
])

df=pd.DataFrame({0:{'SOM':arr}, 1:{'SOM':arr}, 2:{'SOM':arr}, 3:{'SOM':arr}, }).T


print(concat(df, 'SOM', 0))

