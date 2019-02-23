import turicreate as tc
import numpy as np
from math import sqrt
tc.config.set_runtime_config('TURI_NUM_GPUS', 0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

tc.visualization.set_target('gui')

sales = tc.SFrame('home_data.gl/')

tc.visualization.scatter(sales["sqft_living"], sales["price"], "sqft", "price")

len(sales["sqft_living"]), len(sales["price"])
sales["price"].show()

train_data,test_data = sales.random_split(.8,seed=0)

train_df = train_data.to_dataframe()
test_df = test_data.to_dataframe()

train_df.head()

train_X = train_df[['sqft_living', 'price']]






lin_reg = LinearRegression()





train_df.sqft_living.values.shape[0]
lin_reg.fit(train_df.sqft_living.values.reshape((train_df.sqft_living.values.shape[0], 1)),
            train_df.price.values.reshape((train_df.price.values.shape[0], 1)))
lin_reg.intercept_, lin_reg.coef_



preds = lin_reg.predict(test_df.sqft_living.values.reshape((test_df.sqft_living.values.shape[0], 1)))



preds
preds.shape

type(test_df['price'][0])
type(preds[0])
print("Mean squared error: {}".format(sqrt(mean_squared_error(test_df['price'], preds))))

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(test_df['sqft_living'],test_df['price'],'.',
        test_df['sqft_living'],preds,'-')
