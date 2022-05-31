import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:/Users/bymun/PycharmProjects/dataset/COVID-19 Survey Student Responses.csv')

#Sort Dataset and replace characteristic data with numerical data
df = df[['Time spent on Online Class','Rating of Online Class experience','Time spent on self study','Time spent on social media','Time spent on TV']]
df = df.replace({'Excellent':5, 'Good':4, 'Average':3, 'Poor':2, 'Very poor':1, 'Na':np.NaN})
df = df.replace({'n':0, 'No tv':0, 'N':0,' ':0})
df.dropna(axis = 0, how='any')

#The number of cases is divided into positive and negative parts.
x = df[['Time spent on self study','Time spent on Online Class']]
x1 = df[['Time spent on social media','Time spent on TV']]
x2 = df[['Time spent on self study','Time spent on Online Class','Time spent on social media','Time spent on TV']]
y = df[['Rating of Online Class experience']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, train_size = 0.8, test_size = 0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y, train_size = 0.8, test_size = 0.2)

# In the case of model selection, the reason for the linear regression is to analyze the data 
# relatively accurately by establishing an alternative expression and to derive the result value
model = LinearRegression()
model.fit(x_train, y_train)
model1 = LinearRegression()
model2 = LinearRegression()
model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)

y_predict = model.predict(x_test)
y1_predict = model1.predict(x1_test)
y2_predict = model2.predict(x2_test)

plt.subplot(2,3,1)
plt.scatter(df['Time spent on Online Class'], df['Rating of Online Class experience'])
plt.xlabel('Online Class Time')

plt.subplot(2,3,2)
plt.scatter(df['Time spent on self study'], df['Rating of Online Class experience'])
plt.xlabel('Self Study Time')


plt.subplot(2,3,4)
plt.plot(y2_test, y2_predict, alpha=0.4, color='black')
plt.xlabel('Total Result')

plt.subplot(2,3,5)
plt.plot(y_test, y_predict, alpha=0.4, color='red')
plt.xlabel('Study Time')

plt.subplot(2,3,6)
plt.plot(y1_test, y1_predict, alpha=0.4, color='green')
plt.xlabel('Entertainment Time')

#Dataset relationship estimation and selection process through subplot after model learning is completed

plt.show()

