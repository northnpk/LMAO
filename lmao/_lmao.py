import numpy as np
import pandas as pd
from tqdm import tqdm

class LMAO:
    def __init__(self,
                 docs: pd.DataFrame, 
                 mode: str = "directed",
                 verbose: bool=True,
                 ):
        print(f'Setup complete')