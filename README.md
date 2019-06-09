# FIFA19 Potential Predictor
ML project that given FIFA attributes of a player, predicts its Potential rating.

### DATA
Dataset was downloaded from kaggle and contains all players from FIFA19.

### Learner
Learner uses Random Forest Regressor with accurcies that can be found in 'Accuracy' file. 
Some models, such as knn and linear regression, can be found in 'Learner'. 
RF Regressor was trained with Grid Search and CV. 

### Input
Learner takes 6 arguments and produces the predicted Potential attribute of the player.
Arguments need to be passed as array through 'Main' or in terminal separated by spaces.

Attributes are: 
**Age, Overall, Value (M€), Wage (K€), Skill Moves, Release Clause (M€)**
