
import time
from src.pre_process import load_data, pre_process
from src.learner import model

def main():
    start = time.time()
    data = load_data()
    X, y = pre_process(data)
    model(X, y)
    print("\n\nTook %s seconds" % str(time.time() - start))


if __name__ == "__main__":
    main()
