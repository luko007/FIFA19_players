import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import re
import numpy as np
from sklearn.model_selection import train_test_split

DATA_NAME = 'data.csv'
FEET_TO_CM = re.compile (r"([0-9]+)'([0-9]*\.?[0-9]+)")



def model(X, y):
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 19, shuffle=True)
    clf = svm.SVC (kernel='linear', C=1, random_state=19).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

def load_data():
    # Load all csv's to one file
    # names of classes
    df = pd.read_csv (DATA_NAME, index_col=None)
    return df

def salary_to_number(data):
    if type(data) is str:
        return data.replace('M', '000000').replace('.', '').replace('€', '').replace('K', '000')

def change_feet_to_cm(data):
    if type(data) is str:
        m = FEET_TO_CM.match (data)
        if m == None:
            return float ('NaN')
        else:
            return int (m.group(1)) * 12 + float (m.group (2))

def pre_proccess(data):
    y = data['Potential']
    data.drop(['Potential'], axis=1, inplace=True)

    # Deal with nan ==> 0
    data.fillna(0, inplace=True)
    data.replace('', 0, inplace=True)


    columns_to_delete = ['ID', 'Photo', 'Flag', 'Club Logo', 'Body Type', 'Real Face', 'Name']
    [data.drop([col_to_del], axis=1, inplace=True) for col_to_del in columns_to_delete]

    # Salaries
    data['Value'] = data['Value'].apply(salary_to_number)
    data['Wage'] = data['Wage'].apply(salary_to_number)
    data['Release Clause'] = data['Release Clause'].apply(salary_to_number)

    data['Height'] = data['Height'].apply(change_feet_to_cm)
    data['Weight'] = data['Weight'].apply((lambda x: str (x)[:-3]))

    # ST,RS,LW,...:
    avail_positions = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF','RF', 'RW', 'LAM', 'CAM',
                       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB','LDM', 'CDM',
                       'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    for pos in avail_positions:
        data[pos] = data[pos].apply(lambda x: int(x.split('+')[0]) + int(x.split('+')[1]) if type(x) is str else x)

    # data['Joined'] = data['Joined'].apply(lambda x: '0' if x==np.nan else int(str (x)[-4:]))
    data.drop(['Joined'], axis=1, inplace=True)
    data.drop(['Contract Valid Until'], axis=1, inplace=True)
    data.drop(['Loaned From'], axis=1, inplace=True)
    # data['Contract Valid Until'] = data['Contract Valid Until'].apply(lambda x: int(str (x)[-4:]))

    # Dummies
    # Position
    natio_dummy = pd.get_dummies(data['Position'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Position'], axis=1, inplace=True)
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

    # Work Rate ???
    natio_dummy = pd.get_dummies(data['Work Rate'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Work Rate'], axis=1, inplace=True)
    data.replace('', 0.0, inplace=True)
    data = np.nan_to_num(data)
    return data, y


def main():
    data = load_data()
    X, y = pre_proccess(data)
    model(X, y)


if __name__ == "__main__":
    main()