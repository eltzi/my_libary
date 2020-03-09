import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sn
import statsmodels.discrete.discrete_model as sm


path = '/Users/elenitziaferi/PycharmProjects/udemy_ml_logistic_lda_knn/Files/House-Price.csv'
df = pd.read_csv(path)
X = sn.add_constant(df['room_num'])
lm = sn.OLS(df['price'], X).fit()
print(lm.summary())

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = df['price']
X = df[['room_num']]
lr.fit(X, y)
print('intercept',lr.intercept_, 'coef of x', lr.coef_)
print(lr.predict(X))
sns.jointplot(x=df['room_num'], y=df['price'], data=df, kind= 'reg')
plt.show()
sns.jointplot(x=df['room_num'], y=df['price'], data=df, kind= 'resid')
plt.show()

X_multi = df.drop('price', axis=1)
y_multi = df['price']
X_multi_cons = sn.add_constant(X_multi)
lm_multi = sn.OLS(y_multi, X_multi_cons).fit()
print(lm_multi.summary())

lm3 = LinearRegression()
lm3.fit(X_multi, y_multi)
print('intercept',lm3.intercept_, 'coef of x', lm3.coef_)








