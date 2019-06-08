
import numpy as np
import pandas as pd
import re

DATA_NAME = 'data.csv'
FEET_TO_CM = re.compile (r"([0-9]+)'([0-9]*\.?[0-9]+)")


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

def pre_process(data):
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
    all_ratings = ('Crossing,Finishing,HeadingAccuracy,ShortPassing,Volleys,Dribbling,'
                                   'Curve,FKAccuracy,LongPassing,BallControl,Acceleration,SprintSpeed,'
                                   'Agility,Reactions,Balance,ShotPower,Jumping,Stamina,Strength,LongShots,'
                                   'Aggression,Interceptions,Positioning,Vision,Penalties,Composure,Marking,'
                                   'StandingTackle,SlidingTackle,GKDiving,GKHandling,GKKicking,GKPositioning'
                                   ',GKReflexes'.split)(',')
    [data.drop([col_to_del], axis=1, inplace=True) for col_to_del in avail_positions]
    [data.drop([col_to_del], axis=1, inplace=True) for col_to_del in all_ratings]
    # for pos in avail_positions:
    #     new_plus_pos_name = 'PLUS_FOR_'+pos
    #     data[pos] = data[pos].apply(
    #         lambda x: (float(x.split('+')[0]) + float(x.split('+')[1])) if type(x) is str else x)

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

def fix_arrays(data):
    data.replace('', 0.0, inplace=True)
    data.replace(np.nan, 0.0, inplace=True)
