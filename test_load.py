import warnings; warnings.filterwarnings('ignore'); import os, sys, pandas as pd; sys.path.append(os.getcwd()); from models.loader import load_models; m = load_models(); print(f'Keys: {m.keys()}')
