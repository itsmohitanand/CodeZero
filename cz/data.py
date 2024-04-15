from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Preprocess:
    def __init__(self, train_data, val_data, test_data, deg_seasonality_year=2, deg_seasonality_week=2, deg_trend=2):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.add_seasonality_year_col(deg=deg_seasonality_year)
        self.add_seasonality_week_col(deg=deg_seasonality_week)
        self.add_seasonality_day_col(deg=deg_seasonality_year)


        self.add_trend_col(deg=deg_trend)
        self.generate_y()
        
    def _seasonality_year_model(self, poly):
        x = self.train_data.filter(["sin_MOY", "cos_MOY"]).values
        y = self.train_data.Temperature.values
       
        reg = LinearRegression().fit(poly.fit_transform(x), y)
        
        return reg
    
    def _seasonality_week_model(self, poly):
        x = self.train_data.filter(["sin_MOW", "cos_MOW"]).values
        y = self.train_data.Temperature.values - self.train_data.seasonality_year.values
       
        reg = LinearRegression().fit(poly.fit_transform(x), y)
        
        return reg

    def _seasonality_day_model(self, poly):
        x = self.train_data.filter(["sin_MOD", "cos_MOD"]).values
        y = self.train_data.Temperature.values - self.train_data.seasonality_year.values - - self.train_data.seasonality_week.values
       
        reg = LinearRegression().fit(poly.fit_transform(x), y)
        
        return reg
    
    def _trend_model(self, poly):
        # Trend is only computed after removing seasonality_year
        # We want are trend model to be more flexible

        x = self.train_data.filter(["j_day"]).values
        y = (self.train_data.Temperature - self.train_data.seasonality_year).values

        reg = LinearRegression().fit(poly.fit_transform(x), y)
        
        return reg
    
    def add_seasonality_year_col(self, deg):
        poly = PolynomialFeatures(deg)
        reg = self._seasonality_year_model(poly)

        x_train = self.train_data.filter(["sin_MOY", "cos_MOY"])
        self.train_data["seasonality_year"] = reg.predict(poly.fit_transform(x_train))

        x_val = self.val_data.filter(["sin_MOY", "cos_MOY"])
        self.val_data["seasonality_year"] = reg.predict(poly.fit_transform(x_val))

        x_test = self.test_data.filter(["sin_MOY", "cos_MOY"])
        self.test_data["seasonality_year"] = reg.predict(poly.fit_transform(x_test))

    def add_seasonality_week_col(self, deg):
        poly = PolynomialFeatures(deg)
        reg = self._seasonality_week_model(poly)

        x_train = self.train_data.filter(["sin_MOW", "cos_MOW"])
        self.train_data["seasonality_week"] = reg.predict(poly.fit_transform(x_train))

        x_val = self.val_data.filter(["sin_MOW", "cos_MOW"])
        self.val_data["seasonality_week"] = reg.predict(poly.fit_transform(x_val))

        x_test = self.test_data.filter(["sin_MOW", "cos_MOW"])
        self.test_data["seasonality_week"] = reg.predict(poly.fit_transform(x_test))

    def add_seasonality_day_col(self, deg):
        poly = PolynomialFeatures(deg)
        reg = self._seasonality_day_model(poly)

        x_train = self.train_data.filter(["sin_MOD", "cos_MOD"])
        self.train_data["seasonality_day"] = reg.predict(poly.fit_transform(x_train))

        x_val = self.val_data.filter(["sin_MOD", "cos_MOD"])
        self.val_data["seasonality_day"] = reg.predict(poly.fit_transform(x_val))

        x_test = self.test_data.filter(["sin_MOD", "cos_MOD"])
        self.test_data["seasonality_day"] = reg.predict(poly.fit_transform(x_test))

    def add_trend_col(self, deg):

        poly = PolynomialFeatures(deg)
        reg = self._trend_model(poly)

        x_train = self.train_data.filter(["j_day"]).values
        self.train_data["trend"] = reg.predict(poly.fit_transform(x_train))

        x_val = self.val_data.filter(["j_day"]).values
        self.val_data["trend"] = reg.predict(poly.fit_transform(x_val))

        x_test = self.test_data.filter(["j_day"]).values
        self.test_data["trend"] = reg.predict(poly.fit_transform(x_test))

    def generate_y(self, ):
        # remove trend and seasonality_year from y

        self.train_data["y"] = self.train_data.Temperature - self.train_data.seasonality_year - self.train_data.seasonality_week - self.train_data.seasonality_day
        self.val_data["y"] = self.val_data.Temperature - self.val_data.seasonality_year - self.val_data.seasonality_week -  self.val_data.seasonality_day

