
import time
from src.PreProcess import load_data, pre_process
from src.Learner import Learner
import pickle
import numpy as np
import os
import argparse
from pathlib import Path


LEARNER_OBJ_NAME = 'learner.obj'
ATTRIBUTES = ['Age', 'Overall', 'Value', 'Wage', 'Skill Moves', 'Release Clause']


def main():
    learner = Learner()

    # If learner is not saved
    if not Path(LEARNER_OBJ_NAME).is_file():
        print("Fitting model")
        data = load_data()
        X, y = pre_process(data)
        learner.model(X, y)
        save_model(learner)

    # Input example: np.array([18.0, 90, 47000000.0, 41000.0, 4.0, 0.0]).reshape(1,-1)

    player = handle_input()
    player_potential = predict(np.array(player).reshape(1, -1))
    print_result(player, player_potential)


def print_result(player, player_potential):
    all_attributes = ['Potential'] + ['***'] + ATTRIBUTES
    player_att = [player_potential] + ['***'] + player
    row_format = "{:>15}" * (len (all_attributes) + 1)
    print (row_format.format ("", *all_attributes))
    for team, row in zip (all_attributes, [player_att]):
        print (row_format.format ('', *row))


def handle_input():
    parser = argparse.ArgumentParser(description='Predicting FIFA Rating of given player.'
                                                 '\nPlease insert The following attributes separated by spaces:'
                                                 'Age, Overall, Value, Wage, Skill Moves, Release Clause')

    [parser.add_argument(att) for att in ATTRIBUTES]
    args = vars(parser.parse_args())
    new_player = [args[att] for att in ATTRIBUTES]
    return new_player


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
    X need to have the following attributes:
    Age, Overall, Value, Wage, Skill Moves, Release Clause
    """
    learner = load_model()
    return int(learner.get_model().predict(X)**(1/3))


if __name__ == "__main__":
    main()


