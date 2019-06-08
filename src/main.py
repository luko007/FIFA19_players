import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import re
import numpy as np
from sklearn.model_selection import train_test_split

DATA_NAME = 'data.csv'
FEET_TO_CM = re.compile (r"([0-9]+)'([0-9]*\.?[0-9]+)")



def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 19, shuffle=True)
    fix_arrays(X_train)
    fix_arrays(X_test)

    # [fix_arrays(arr) for arr in [X_train, X_test]]

    # TEST = (X[int(X.shape[0]*0.72):int(X.shape[0]*0.73)])
    # TEST.replace('', 0.0, inplace=True)
    # TEST.replace(np.nan, 0.0, inplace=True)
    # nume = []
    # for a in TEST.values:
    #     for b in a:
    #         nume.append(isnumeric(b))
    # a = [a for a in X_test.values]
    # b = [b for b in a]
    # print([isnumeric(s) for s in b])

    print("SVM with SGD")
    # fit_and_predict(X_train, y_train, X_test, y_test,
    #                 SGDClassifier (loss='hinge', random_state=41, n_jobs=-1,
    #                max_iter=1000, tol=None))
    print("SVM Linear")
    # fit_and_predict(X_train, y_train, X_test, y_test,
    #                svm.LinearSVC(multi_class='ovr', C=1e5))
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    # k=1 is the best
    print("KNN")
    param_grid = {'n_neighbors': np.arange (1, 30)}
    knn_cv = GridSearchCV (KNeighborsClassifier(), param_grid, cv=5)
    knn_cv.fit (X_train, y_train)
    print(knn_cv.best_params_)
    print(knn_cv.best_score_)

    fit_and_predict(X_train, y_train, X_test, y_test,
                    KNeighborsClassifier(n_neighbors=15))
    print("GaussianNB")
    fit_and_predict(X_train, y_train, X_test, y_test,
                    GaussianNB())
    print("Logistic")
    # fit_and_predict(X_train, y_train, X_test, y_test,
    #                 LogisticRegression(multi_class='ovr', solver='saga', max_iter=10000, n_jobs=-1))


def fit_and_predict(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print('accuracy %s' % accuracy)
    return accuracy


def fix_arrays(data):
    data.replace('', 0.0, inplace=True)
    data.replace(np.nan, 0.0, inplace=True)



def load_data():
    # Load all csv's to one file
    # names of classes
    df = pd.read_csv (DATA_NAME, index_col=None)
    return df

def salary_to_number(data):
    if type(data) is str:
        return data.replace('M', '000000').replace('.', '').replace('€', '').replace('K', '000')
    return 0.0

def change_feet_to_cm(data):
    if type(data) is str:
        m = FEET_TO_CM.match (data)
        if m == None:
            return 0.0
        else:
            return int (m.group(1)) * 12 + float (m.group (2))

def pre_proccess(data):
    data.replace(np.nan, 0.0, inplace=True)
    data.replace('', 0.0, inplace=True)
    y = data['Potential']
    data.drop(['Potential'], axis=1, inplace=True)

    # Deal with nan ==> 0
    # data.fillna(0, inplace=True)
    # data.replace('', 0, inplace=True)

    columns_to_delete = ['ID', 'Photo', 'Flag', 'Club Logo', 'Body Type', 'Real Face', 'Name']
    [data.drop([col_to_del], axis=1, inplace=True) for col_to_del in columns_to_delete]

    # Salaries
    data['Value'] = data['Value'].apply(salary_to_number)
    data['Wage'] = data['Wage'].apply(salary_to_number)
    data['Release Clause'] = data['Release Clause'].apply(salary_to_number)

    data['Height'] = data['Height'].apply(change_feet_to_cm)
    data['Weight'] = data['Weight'].apply((lambda x: str(x)[:-3]))

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

    # Dummies - One Hot Encoding

    # Position
    # natio_dummy = pd.get_dummies(data['Position'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Position'], axis=1, inplace=True)
    # natio_dummy = pd.get_dummies(data['Nationality'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Nationality'], axis=1, inplace=True)
    # Club
    # natio_dummy = pd.get_dummies(data['Club'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Club'], axis=1, inplace=True)
    # Preferred Foot
    natio_dummy = pd.get_dummies(data['Preferred Foot'])
    data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Preferred Foot'], axis=1, inplace=True)

    # Work Rate ???
    # natio_dummy = pd.get_dummies(data['Work Rate'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Work Rate'], axis=1, inplace=True)
    data = data.reset_index(drop=True)

    return data, y


def main():
    data = load_data()
    X, y = pre_proccess(data)
    model(X, y)


if __name__ == "__main__":
    main()