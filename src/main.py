
import time
from src.pre_process import load_data, pre_process
from src.learner import model
import pickle

LEARNER_OBJ_NAME = "learner.obj"

def main():
    start = time.time()
    data = load_data()
    X, y = pre_process(data)
    learner = model(X, y)
    save_model(learner)
    print("\n\nTook %s seconds" % str(time.time() - start))


def save_model(learner):
    with open(LEARNER_OBJ_NAME, 'w') as filehandler:
        pickle.dump(learner, filehandler)


def load_model():
    with open(LEARNER_OBJ_NAME, 'w') as learner_obj:
        learner = pickle.load(learner_obj)
    return learner


def predict(X):
    learner = load_data()
    return learner.predict(X)

if __name__ == "__main__":
    main()
