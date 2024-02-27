import numpy as np 
import pandas as pd 

import wandb 
wandb.login()

def query_runs():
    api = wandb.Api()
