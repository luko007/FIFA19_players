import pandas as pd
import glob
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import model_selection
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

DATA_NAME = 'data.csv'

def load_data():
    # Load all csv's to one file
    # names of classes
    df = pd.read_csv (DATA_NAME, index_col=None)
    return df

def salary_to_number(data):
    if type(data) is str:
        return data.replace('M', '000000').replace('.', '').replace('€', '').replace('K', '000')

feet_to_cm = re.compile (r"([0-9]+)'([0-9]*\.?[0-9]+)")
def change_feet_to_cm(data):
    if type(data) is str:
        m = feet_to_cm.match (data)
        if m == None:
            return float ('NaN')
        else:
            return int (m.group(1)) * 12 + float (m.group (2))



def pre_proccess(data):
    columns_to_delete = ['ID', 'Photo', 'Flag', 'Club Logo', 'Body Type', 'Real Face', 'Name']
    [data.drop([col_to_del], axis=1, inplace=True) for col_to_del in columns_to_delete]

    # Salaries
    data['Value'] = data['Value'].apply(salary_to_number)
    data['Wage'] = data['Wage'].apply(salary_to_number)
    data['Release Clause'] = data['Release Clause'].apply(salary_to_number)

    data['Height'] = data['Height'].apply(change_feet_to_cm)
    data['Weight'] = data['Weight'].apply((lambda x: str (x)[:-3]))

    # data['Joined'] = data['Joined'].apply(lambda x: '0' if x==np.nan else int(str (x)[-4:]))
    data.drop(['Joined'], axis=1, inplace=True)
    data.drop(['Contract Valid Until'], axis=1, inplace=True)
    data.drop(['Loaned From'], axis=1, inplace=True)
    # data['Contract Valid Until'] = data['Contract Valid Until'].apply(lambda x: int(str (x)[-4:]))

    # Position
    natio_dummy = pd.get_dummies(data['Position'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Position'], axis=1, inplace=True)

    # Dummies
    natio_dummy = pd.get_dummies(data['Nationality'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Nationality'], axis=1, inplace=True)
    # Club
    natio_dummy = pd.get_dummies(data['Club'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Club'], axis=1, inplace=True)
    # Preferred Foot
    natio_dummy = pd.get_dummies(data['Preferred Foot'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Preferred Foot'], axis=1, inplace=True)
    # Position
    natio_dummy = pd.get_dummies(data['Position'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Position'], axis=1, inplace=True)

    # Work Rate ???
    natio_dummy = pd.get_dummies(data['Work Rate'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Work Rate'], axis=1, inplace=True)



    print(data.head())


def main():
    data = load_data()
    pre_proccess(data)


if __name__ == "__main__":
    main()