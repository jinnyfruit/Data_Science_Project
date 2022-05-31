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

    # turn column into numerical data and change dirty data into NAN and drop it
    print("\n-------------------- Before convert & drop --------------------")
    print(df['Time spent on TV'])
    df = (df.drop('Time spent on TV', axis=1).join(df['Time spent on TV'].apply(pd.to_numeric, errors='coerce')))
    df.dropna(inplace=True)
    print("\n-------------------- After convert & drop --------------------")
    print(df['Time spent on TV'])

    # change categorical data into numerical data
    df = df.replace({'Increased': 50, 'Decreased': -50, 'Remain Constant': 0})
    df = df.replace({'NO': 0, 'YES': 50})

    # data split into train/test data - 'Time spent on TV' 'Time spent on social media'
    X = df[['Time spent on fitness', 'Time spent on sleep', 'Number of meals per day', 'Change in your weight']]
    y = df['Health issue during lockdown'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

    # Standardization 평균 = 0 / 표준편차 = 1
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
    # sns.clustermap(X.corr(), annot=True, cmap='viridis')
    plt.show()

# read the data
df = pd.read_csv('covid_student.csv')
print("-------------------- Read the data --------------------\n")
print(df)

TreeClassifier(df)
