import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, ds_path = "data/"):
        shuffle=False # This is important to predict out of time 
        self.data = pd.read_csv(ds_path+"train.csv")
        self.train_data, self.val_data = train_test_split(self.data, test_size=0.2, shuffle=shuffle)
        self.test_data = pd.read_csv(ds_path+"test.csv")
        self.sample_submission = pd.read_csv(ds_path+"sample_submission.csv")
        self.seed = 42
    # CHANGED
    def k_fold(self, n_splits = 5):
        return KFold(n_splits, shuffle=True, random_state = self.seed)