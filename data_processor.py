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
        self.data["marital_status"] = self.data["marital_status"].replace(['Divorced', 'Separated', 'Widowed'], 'Single')
        self.data["marital_status"] = self.data["marital_status"].replace(['Married-civ-spouse', 'Married-spouse-absent',
                                                                           'Married-AF-spouse'], 'Married')
        self.data["relationship"] = self.data["relationship"].replace(['Not-in-family', 'Other-relative'], 'Separated')
        self.data["relationship"] = self.data["relationship"].replace(['Husband', 'Wife'], 'Married')
        self.data["relationship"] = self.data["relationship"].replace(['Unmarried', 'Own-child'], 'Single')
        self.data["workclass"] = self.data["workclass"].replace(['Self-emp-not-inc', 'Local-gov', "State-gov", 
                                                                 "Self-emp-inc", "Federal-gov", "Without-pay", 
                                                                 "Never-worked", '<=50K'], 'govermental')
        self.data['occupation'].replace(to_replace=['Machine-op-inspct', 'Farming-fishing', 'Craft-repair', 
                                                    'Transport-moving', 'Handlers-cleaners'], value="Blue_collar", 
                                        inplace=True)
        self.data['occupation'].replace(to_replace=['Adm-clerical', 'Tech-support', 'Exec-managerial', 
                                                    'Prof-specialty'], value="White_collar", inplace=True)
        self.data['occupation'].replace(to_replace=['Protective-serv', 'Armed-Forces'], 
                                        value="Brown_collar/Protective_service", inplace=True)
        self.data['occupation'].replace(to_replace=['Other-service', 'Sales', 'Priv-house-serv'], 
                                        value="Pink_collar/Service_and_sales", inplace=True)
        self.data["education"] = self.data["education"].replace(['Prof-school', "Assoc-acdm", "Assoc-voc"], 'high-school')
        self.data["education"] = self.data["education"].replace(['Some-college', 'Doctorate', 'Bachelors', "Masters"], 
                                                                'college')
        self.data["education"] = self.data["education"].replace(['7th-8th', '10th', '11th', "1st-4th", "5th-6th", 
                                                                "12th", "9th", "Preschool"], 'pre-hs')
        return self.data