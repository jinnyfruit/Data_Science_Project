import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


pd.set_option('display.max_columns', None)
df = pd.read_csv('/Users/jeongdeok/Downloads/COVID-19 Survey Student Responses.csv')
df = df.replace({'n':0, 'No tv':0, 'N':0,' ':0})
#print(df.isnull().any())
df.dropna(axis = 0, inplace=True)


X3 = df[['Age of Subject', 'Time spent on fitness', 'Time spent on sleep', 'Time spent on social media', 'Time spent on TV']]
Y3 = df[['Health issue during lockdown']]
train_X, test_X, train_Y, test_Y = train_test_split(X3, Y3, test_size=0.2)
sub_input, val_input, sub_target, val_target = train_test_split(train_X, train_Y, test_size=0.2)


params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
dt = DecisionTreeClassifier()

gs = GridSearchCV(dt, params, n_jobs=-1)
gs.fit(sub_input, sub_target)

print(gs.best_score_)
print(gs.best_params_)
print(gs.score(test_X, test_Y))
"""
score = []
feature_importance = []
max_depth = [3, 4, 5, 6, 7]

for i in max_depth:
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(train_X, train_Y)

    plt.figure(figsize=(10,7))
    score.append(dt.score(test_X,test_Y))
    feature_importance.append(dt.feature_importances_)
    plot_tree(dt)
    plt.show()


for i in range(5):
    print('max_depth:', max_depth[i], 'score:', score[i],'feature_importance:', feature_importance[i])

"""