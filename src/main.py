
import time
from src.pre_process import load_data, pre_process
from src.learner import model
import pickle
import numpy as np
import pandas as pd

LEARNER_OBJ_NAME = "learner.obj"

def main():
    start = time.time()
    data = load_data()
    X, y = pre_process(data)
    learner = model(X, y)
    save_model(learner)
    print("\n\nTook %s seconds" % str(time.time() - start))


def save_model(learner):
    with open(LEARNER_OBJ_NAME, 'wb') as output:
        pickle.dump(learner, output)


def load_model():
    with open(LEARNER_OBJ_NAME, 'w') as learner_obj:
        learner = pickle.load(learner_obj)
    return learner


def predict(X):
    cleaned_X = pre_process(X)[0]
    learner = load_data()
    return learner.predict(X)

if __name__ == "__main__":
    # main()
    header = pd.read_csv('header_without_y.csv', header=None)
    player = pd.read_csv('input.csv', header=header)
    predict(player)
