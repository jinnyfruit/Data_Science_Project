'''
file name : Term Project - Regression of of online learning adaptability according to online experience
name: 202035513 Ji Woo Kim
modified: 2022.05.01

'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

def TreeClassifier (df):
    # check dataframe
    print("\n-------------------- Data describe --------------------")
    print(df.describe())

    # Data preprocessing
    # find NAN and drop it
    print("\n-------------------- Find NAN --------------------")
    print(df.isna())

    print("\n-------------------- Drop NAN and show --------------------")
    df.dropna(inplace=True)
    print(df)

    # change categorical data into numerical data
    df = df.replace({'On (All the time)': 100, 'Always On': 100, 'On (Always On)': 100, 'On (To answer Teacher)': 50,
                     'On (When teacher asks)': 50, 'Off (Untidy appearance)': 0, 'Off (Shy to switch on)': 0,
                     'Off (No mic available)': 50, 'Off (No webcam available)': 50, 'Off (Other reason)': -50,
                     'Off (Other reason)': -50, 'Off (Do not want to reply)': -50})
    df = df.replace({'No': -50, 'Yes': 50})
    df.dropna(inplace=True)

    score = []
    feature_importance = []
    max_depth = [3, 4, 5, 6, 7]

    X = df[['Webcam status during class', 'Mic status during class', 'Are you interested in attending online classes?']]
    y = df['Are you able to understand the concepts through online classes?'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

    # Standardization
    scaler = StandardScaler()
    scaler = scaler.fit_transform(X)

    print("\n-------------------- X print --------------------")
    print(X)
    print("\n-------------------- y print --------------------")
    print(y)
    print("\n-------------------- X - train print --------------------")
    print(X_train)
    print("\n-------------------- X - test print --------------------")
    print(X_test)
    print("\n-------------------- y - train print --------------------")
    print(y_train)
    print("\n-------------------- y - test print --------------------")
    print(y_test)

    # make a model and train model
    tree_model = tree.DecisionTreeClassifier()
    tree_model = tree_model.fit(X_train, y_train)

    # test a model
    y_predict = tree_model.predict(X_train)

    # model accuracy
    print("\n-------------------- Decision Tree Accuracy --------------------")
    print(round(tree_model.score(X_test, y_test), 3))

    # feature importance scoring

    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    # Show Tree
    tree.plot_tree(tree_model)

    # Shows the relationship between features
    # sns.clustermap(X.corr(), annot=True, cmap='viridis')'''
    plt.show()

# read the data
df = pd.read_csv('covid_students_survay.csv')
print("-------------------- Read the data --------------------\n")
print(df)

TreeClassifier(df)

