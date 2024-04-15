import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np

class Data:
    def __init__(self, ds_path = "data/", clean=False):
        shuffle=False # This is important to predict out of time 
        self.data = pd.read_csv(ds_path+"train.csv")
        self.test_data = pd.read_csv(ds_path+"test.csv")
        if clean:
            self._clean_data()
        self.preprocess_data()
        
        self._add_minuite_of_the_year()
        self.train_data, self.val_data = train_test_split(self.data, test_size=0.2, shuffle=shuffle)
        
        self.sample_submission = pd.read_csv(ds_path+"sample_submission.csv")
        self.seed = 42

    def k_fold(self, n_splits = 5):
        return KFold(n_splits, shuffle=True, random_state = self.seed)
    
    def _add_minuite_of_the_year(self,):
        
        self.data["sin_MOY"] = self.data.date.apply(self.sin_MOY)
        self.data["cos_MOY"] = self.data.date.apply(self.cos_MOY)

        self.test_data["sin_MOY"] = self.test_data.date.apply(self.sin_MOY)
        self.test_data["cos_MOY"] = self.test_data.date.apply(self.cos_MOY)

        return None

    def sin_MOY(self, x):
        
        min_x = self._min_encoding(x)
        sin_day = np.sin(min_x)
        
        return sin_day

    def cos_MOY(self, x):
        
        min_x = self._min_encoding(x)
        cos_day = np.cos(min_x)

        return cos_day


    def _min_encoding(self, x):
        
        day = x.timetuple().tm_yday 
        min_of_the_year = 24*60*day + 60*x.hour + x.minute
        max_min = 366*24*60

        return (min_of_the_year*2*np.pi)/max_min

    def _str_to_datetime(self, x):
        x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return x
    
    def _to_julian_day(self, x):

        return x.to_julian_date()
        

    def generate_submission(self, proba_preds):
        assert proba_preds.shape==(1000, 5360)

        quantiles = [0.025,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975]
        quantile_predictions = {"id": self.test_data.id.values}
        quantile_predictions.update({q: None for q in quantiles})
        for q in quantiles:
            quantile_predictions[q] = np.quantile(proba_preds, q, axis=0)
        submission_df = pd.DataFrame.from_dict(quantile_predictions)
        submission_df.head(2)

        return submission_df

    def preprocess_data(self, ):

        self.data.date = self.data.date.apply(self._str_to_datetime)
        self.data["j_day"] = self.data.date.apply(self._to_julian_day)

        self.test_data.date = self.test_data.date.apply(self._str_to_datetime)
        self.test_data["j_date"] = self.test_data.date.apply(self._to_julian_day)

    def _clean_data(self):
        index = np.where(self.data["feature_CB"] == 0)[0]
        init_samples = self.data.shape[0]
        self.data = self.data.drop(index, axis="index")
        print(F"Sample size changed from {init_samples} to {self.data.shape[0]}")
