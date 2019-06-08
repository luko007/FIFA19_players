import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error, median_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import time
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

    # svm_learner (X_test, X_train, y_test, y_train)

    # linear_reg (X_test, X_train, y_test, y_train)

    random_forest (X_test, X_train, y_test, y_train)

    # lasso (X_test, X_train, y_test, y_train)

    # knn(X_test, X_train, y_test, y_train)


def svm_learner(X_test, X_train, y_test, y_train):
    print ("SVM with SGD")
    fit_and_predict (X_train, y_train, X_test, y_test,
                     SGDClassifier (loss='hinge', random_state=41, n_jobs=-1,
                                    max_iter=1000, tol=None))
    print ("Linear SVM")
    lin_svc = LinearSVC (tol=1e-5, random_state=19).fit (X_train, y_train)
    y_pred = lin_svc.predict (X_test)
    regressor_scoring (y_pred, y_test)


def linear_reg(X_test, X_train, y_test, y_train):
    print ("Linear Regression")
    # normalize=True ?
    lin_reg = LinearRegression ().fit (X_train, y_train)
    y_pred = lin_reg.predict (X_test)
    regressor_scoring (y_pred, y_test, "Linear Regression")


def lasso(X_test, X_train, y_test, y_train):
    print ("\nLasso")
    ridge = LassoCV (eps=1, alphas=[1e-2, 1e-3, 1e-4, 1e-5], tol=0.1, cv=5).fit (X_train, y_train)
    y_pred = ridge.predict (X_test)
    regressor_scoring (y_pred, y_test, "Lasso")


def random_forest(X_test, X_train, y_test, y_train):
    """Results were:
    {'max_depth': 15, 'n_estimators': 500}
    Random Forest MSE is: 1.39288676662
    Random Forest Max error is: 10.322
    Random Forest Median error of: 0.438544869642
    """
    print ("\nRandom Forest Regressor")
    OPTIMAL_DEPTH = 15
    OPTIMAL_EST = 500
    regr = RandomForestRegressor (max_depth=OPTIMAL_DEPTH,
                                  n_estimators=OPTIMAL_EST,
                                  random_state=18, )
    # regr = AdaBoostRegressor (regr, n_estimators=300, random_state=19)
    regr.fit(X_train, y_train)
    y_pred = regr.predict (X_test)
    regressor_scoring(y_pred, y_test, "Random Forest")


def grid_search_random_reg(X_train, y_train):
    gs_random_reg = GridSearchCV (
        estimator=RandomForestRegressor (),
        param_grid={
            'max_depth': [3, 9, 15, 25],
            'n_estimators': (150, 100, 500, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result = gs_random_reg.fit (X_train, y_train)
    best__random_reg_params = grid_result.best_params_
    print (best__random_reg_params)
    return best__random_reg_params

def regressor_scoring(y_pred, y_test, name):
    print (name+" MSE is: %s" % mean_squared_error (y_test, y_pred))
    print (name+" Max error is: %s" % max_error (y_test, y_pred))
    print (name+" Median error of: %s" % median_absolute_error (y_test, y_pred))

def knn(X_test, X_train, y_test, y_train):
    # k=1 is the best
    print ("KNN")
    param_grid = {'n_neighbors': np.arange (1, 30)}
    knn_cv = GridSearchCV (KNeighborsClassifier (), param_grid, cv=5)
    knn_cv.fit (X_train, y_train)
    print (knn_cv.best_params_)
    print (knn_cv.best_score_)
    fit_and_predict (X_train, y_train, X_test, y_test,
                     KNeighborsClassifier (n_neighbors=1))

def fit_and_predict(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print('accuracy %s' % accuracy)
    return accuracy, y_pred

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

    columns_to_delete = ['ID', 'Photo', 'Flag', 'Club Logo', 'Body Type', 'Real Face', 'Name', 'Jersey Number']
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
        new_plus_pos_name = 'PLUS_FOR_'+pos
        data[pos] = data[pos].apply(
            lambda x: (float(x.split('+')[0]) + float(x.split('+')[1])) if type(x) is str else x)

    # update_position_rating (avail_positions, data)

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
    # natio_dummy = pd.get_dummies(data['Preferred Foot'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Preferred Foot'], axis=1, inplace=True)

    # Work Rate ???
    # natio_dummy = pd.get_dummies(data['Work Rate'])
    # data = pd.concat([data, natio_dummy], axis=1)
    data.drop(['Work Rate'], axis=1, inplace=True)

    data = data.reset_index(drop=True)

    return data, y


def update_position_rating(avail_positions, data):
    for pos in avail_positions:
        new_plus_pos_name = 'PLUS_FOR_' + pos
        data[new_plus_pos_name] = data[pos].apply (
            lambda x: float (x.split ('+')[1]) if type (x) is str else x)
        data[pos] = data[pos].apply (lambda x: float (x.split ('+')[0]) if type (x) is str else x)


def main():
    data = load_data()
    X, y = pre_proccess(data)
    model(X, y)


if __name__ == "__main__":
    start = time.time()
    main()
    print("\n\nTook %s seconds" % str(time.time() - start))
