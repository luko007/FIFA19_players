from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error, median_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split

from src.pre_process import fix_arrays

def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 19, shuffle=True)
    fix_arrays(X_train)
    fix_arrays(X_test)

    # svm_learner (X_test, X_train, y_test, y_train)

    # linear_reg (X_test, X_train, y_test, y_train)

    learner = random_forest (X_test, X_train, y_test, y_train)

    # lasso (X_test, X_train, y_test, y_train)

    # knn(X_test, X_train, y_test, y_train)

    return learner


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
    lin_reg = LinearRegression ().fit (X_train, y_train)
    y_pred = lin_reg.predict (X_test)
    regressor_scoring (y_pred, y_test, "Linear Regression")


def lasso(X_test, X_train, y_test, y_train):
    print ("\nLasso")
    ridge = LassoCV (eps=1, alphas=[1e-2, 1e-3, 1e-4, 1e-5], tol=0.1, cv=5).fit (X_train, y_train)
    y_pred = ridge.predict (X_test)
    regressor_scoring (y_pred, y_test, "Lasso")


def random_forest(X_test, X_train, y_test, y_train):
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
    return regr


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


