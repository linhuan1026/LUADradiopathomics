import pandas as pd
import numpy as np


def save_results_to_pandas(data, save_path=None):

    df = pd.DataFrame(data=data)
    if save_path != None:
        df.to_csv(save_path, index=False)
    return df