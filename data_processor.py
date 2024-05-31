import pandas as pd
from scipy import stats

class DataProcessor:
    def __init__(self, data_url, column_names, z_score_threshold):
        self.data_url = data_url
        self.column_names = column_names
        self.z_score_threshold = z_score_threshold
        self.data = self.load_data()
        
    def load_data(self):
        data = pd.read_csv(self.data_url, names=self.column_names, sep=",\s*", engine="python")
        return data
    
    def clean_data(self):
        self.data.replace('?', pd.NA, inplace=True)
        self.data['workclass'].fillna(self.data['workclass'].mode()[0], inplace=True)
        self.data['native_country'].fillna(self.data['native_country'].mode()[0], inplace=True)
        self.data['occupation'].fillna('Other-service', inplace=True)
        self.data.drop_duplicates(keep='first', inplace=True, ignore_index=True)
        return self.data
    
    def remove_outliers(self, numerical_variables):
        for variable in numerical_variables:
            z_scores = abs(stats.zscore(self.data[variable]))
            outliers_indices = z_scores[z_scores > self.z_score_threshold].index
            self.data = self.data.drop(outliers_indices)
        self.data.reset_index(drop=True, inplace=True)
        return self.data

    def transform_data(self):
        self.data["marital_status"] = self.data["marital_status"].replace(
            ['Divorced', 'Separated', 'Widowed'], 'Single')
        self.data["marital_status"] = self.data["marital_status"].replace(
            ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
        self.data["relationship"] = self.data["relationship"].replace(
            ['Not-in-family', 'Other-relative'], 'Separated')
        self.data["relationship"] = self.data["relationship"].replace(
            ['Husband', 'Wife'], 'Married')
        self.data["relationship"] = self.data["relationship"].replace(
            ['Unmarried', 'Own-child'], 'Single')
        self.data["workclass"] = self.data["workclass"].replace(
            ['Self-emp-not-inc', 'Local-gov', "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked", '<=50K'], 'govermental')
        self.data['occupation'] = self.data['occupation'].replace(
            {'Machine-op-inspct': 'Blue_collar', 'Farming-fishing': 'Blue_collar', 'Craft-repair': 'Blue_collar', 'Transport-moving': 'Blue_collar', 'Handlers-cleaners': 'Blue_collar'})
        self.data['occupation'] = self.data['occupation'].replace(
            {'Adm-clerical': 'White_collar', 'Tech-support': 'White_collar', 'Exec-managerial': 'White_collar', 'Prof-specialty': 'White_collar'})
        self.data['occupation'] = self.data['occupation'].replace(
            {'Protective-serv': 'Brown_collar/Protective_service', 'Armed-Forces': 'Brown_collar/Protective_service'})
        self.data['occupation'] = self.data['occupation'].replace(
            {'Other-service': 'Pink_collar/Service_and_sales', 'Sales': 'Pink_collar/Service_and_sales', 'Priv-house-serv': 'Pink_collar/Service_and_sales'})
        self.data["education"] = self.data["education"].replace(
            {'Prof-school': 'high-school', "Assoc-acdm": 'high-school', "Assoc-voc": 'high-school'})
        self.data["education"] = self.data["education"].replace(
            {'Some-college': 'college', 'Doctorate': 'college', 'Bachelors': 'college', "Masters": 'college'})
        self.data["education"] = self.data["education"].replace(
            {'7th-8th': 'pre-hs', '10th': 'pre-hs', '11th': 'pre-hs', "1st-4th": 'pre-hs', "5th-6th": 'pre-hs', "12th": 'pre-hs', "9th": 'pre-hs', "Preschool": 'pre-hs'})
        return self.data
