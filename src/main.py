
import time
from src.pre_process import load_data, pre_process
from src.learner import model
import pickle
import numpy as np
import pandas as pd
import os
import math

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
    assert (os.path.getsize(LEARNER_OBJ_NAME) > 0)
    with open(LEARNER_OBJ_NAME, 'rb') as learner_obj:
        learner = pickle.load(learner_obj)
    return learner


def predict(X):
    """
    X need to have the following attributes: Age, Overall, Value, Wage, Skill Moves, Release Clause
    """
    learner = load_model()
    return int(learner.predict(X)**(1/3))

if __name__ == "__main__":
    main()
    player = np.array([18.0, 90, 47000000.0, 41000.0, 4.0, 0.0]).reshape(1,-1)
    print(predict(player))

