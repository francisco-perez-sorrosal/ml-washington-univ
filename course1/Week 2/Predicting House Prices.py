# %% markdown
# # Fire up graphlab create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)
# %%
import turicreate as graphlab
# %%
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.config.set_runtime_config('TURI_NUM_GPUS', 8)
# %% markdown
# # Load some house sales data
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.
# %%
sales = graphlab.SFrame('home_data.gl/')
# %%
sales
# %% markdown
# # Exploring the data for housing sales
# %% markdown
# The house price is correlated with the number of square feet of living space.
# %%
graphlab.visualization.set_target('gui')
len(sales["sqft_living"]), len(sales["price"])
sales["price"].show()
# %%
# sales.show(view="Scatter Plot", x="sqft_living", y="price")
graphlab.visualization.scatter(sales["sqft_living"], sales["price"])
# %% markdown
# # Create a simple regression model of sqft_living to price
# %% markdown
# Split data into training and testing.
# We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).
# %%
train_data,test_data = sales.random_split(.8,seed=0)
# %% markdown
# ## Build the regression model using only sqft_living as a feature
# %%
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)
# %% markdown
# # Evaluate the simple model
# %%
print(test_data['price'].mean())
# %%
print(sqft_model.evaluate(test_data))
# %% markdown
# RMSE of about \$255,170!
# %% markdown
# # Let's show what our predictions look like
# %% markdown
# Matplotlib is a Python plotting library that is also useful for plotting.  You can install it with:
# 'pip install matplotlib'
# %%
import matplotlib.pyplot as plt
%matplotlib inline
# %%

preds = sqft_model.predict(test_data)
preds
plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')
# %% markdown
# Above:  blue dots are original data, green line is the prediction from the simple regression.
#
Below: we can view the learned regression coefficients.
# %%
sqft_model.get('coefficients')
# %% markdown
# # Explore other features in the data
#
To build a more elaborate model, we will explore using more features.
# %%
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
# %%
sales[my_features].show()
# %%
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
# %% markdown
# Pull the bar at the bottom to view more of the data.
#
98039 is the most expensive zip code.
# %% markdown
# # Build a regression model with more features
# %%
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
# %%
print my_features
# %% markdown
# ## Comparing the results of the simple model with adding more features
# %%
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
# %% markdown
# The RMSE goes down from \$255,170 to \$179,508 with more features.
# %% markdown
# # Apply learned models to predict prices of 3 houses
# %% markdown
# The first house we will use is considered an "average" house in Seattle.
# %%
house1 = sales[sales['id']=='5309101200']
# %%
house1
# %% markdown
# <img src="http://info.kingcounty.gov/Assessor/eRealProperty/MediaHandler.aspx?Media=2916871">
# %%
print house1['price']
# %%
print sqft_model.predict(house1)
# %%
print my_features_model.predict(house1)
# %% markdown
# In this case, the model with more features provides a worse prediction than the simpler model with only 1 feature.  However, on average, the model with more features is better.
# %% markdown
# ## Prediction for a second, fancier house
#
We will now examine the predictions for a fancier house.
# %%
house2 = sales[sales['id']=='1925069082']
# %%
house2
# %% markdown
# <img src="https://ssl.cdn-redfin.com/photo/1/bigphoto/302/734302_0.jpg">
# %%
print sqft_model.predict(house2)
# %%
print my_features_model.predict(house2)
# %% markdown
# In this case, the model with more features provides a better prediction.  This behavior is expected here, because this house is more differentiated by features that go beyond its square feet of living space, especially the fact that it's a waterfront house.
# %% markdown
# ## Last house, super fancy
#
Our last house is a very large one owned by a famous Seattleite.
# %%
bill_gates = {'bedrooms':[8],
              'bathrooms':[25],
              'sqft_living':[50000],
              'sqft_lot':[225000],
              'floors':[4],
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}
# %% markdown
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Bill_gates%27_house.jpg/2560px-Bill_gates%27_house.jpg">
# %%
print my_features_model.predict(graphlab.SFrame(bill_gates))
# %% markdown
# The model predicts a price of over $13M for this house! But we expect the house to cost much more.  (There are very few samples in the dataset of houses that are this fancy, so we don't expect the model to capture a perfect prediction here.)
# %%
