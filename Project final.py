
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from operator import itemgetter
dfHistorical = pd.read_csv('Historical_MVPdata.csv')
dfCurrent = pd.read_csv('CurrentMVP.csv')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
dfHistorical.head() 
dfHistorical.describe()   
dfHistorical.info()    
dfHistorical.isnull().sum()
dfHistorical=dfHistorical.dropna()
import seaborn as sns
Historical = dfHistorical[['G', 'Team Wins', 'Overall Seed', 'MP', 'PTS', 'TRB', 'AST', 'STL','BLK', 'FG%', '3P%','FT%', 'WS', 'WS/48', 'VORP', 'BPM']]
Historical.corr()

sns.heatmap(Historical.corr(), cmap="Greens")
train, test = train_test_split(dfHistorical, test_size = 0.3, random_state = 101)

xtrain = train[['3P%', 'STL', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
ytrain = train[['Share']]

xtest = test[['BLK', 'STL', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
ytest = test[['Share']]
            
model=LinearRegression()

model.fit(xtrain,ytrain)

#evaluate the model performance on training data
y_pred=model.predict(xtest)
sns.regplot(x=ytest, y=y_pred, ci=None, color="r")

rmse_test =mean_squared_error(ytest,y_pred,squared=False) 
print(rmse_test)     
#0.4      
from sklearn.ensemble import RandomForestRegressor
rfc=RandomForestRegressor(random_state=1,n_estimators=200)
rfc.fit(xtrain,ytrain.values.ravel())
#evaluate on train
#performace on testing
y_pred=rfc.predict(xtest)
mse = mean_squared_error(ytest, y_pred)
rmse_test =mean_squared_error(ytest,y_pred,squared=False) 
print(rmse_test) 
#0.3
rfc.feature_importances_
plt.barh(xtrain.columns, rfc.feature_importances_)

#KNN

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor


from sklearn.pipeline import Pipeline  
pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsRegressor())])


# The pipeline can be used as any other model
#only difference here is that it also scales the data first here before knn
# and avoids leaking the test set into the train set
pipe.fit(xtrain, ytrain)

y_pred_pipe=pipe.predict(xtest)
mse = mean_squared_error(ytest, y_pred)
rmse_test =mean_squared_error(ytest,y_pred,squared=False) 
print(rmse_test) 
#0.3

####hyperparameter tuning::finding the values of K(no of neighbors) and p(type of distance matrix used)

###doing cross validation using the pipeline

##what values do you want it to check
param_grid = { 'b__n_neighbors': range(1,20), 'b__p': [1,2]} 
#thumb rule: value of K must not be very high when compared to 
#square root of len(x_train)

##import, initialize, fit, get best parameters
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe,param_grid,verbose=3,scoring="f1",cv=10) #initalize 

# May take awhile!
grid.fit(xtrain,ytrain)

# You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best\_estimator_ attribute:

grid.best_params_
# grid.cv_results_

####how is the performance in the test set with tuned parameters

pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsRegressor(n_neighbors=1,p=1))])
pipe.fit(xtrain, ytrain)
y_pred_pipe=pipe.predict(xtest)
mse = mean_squared_error(ytest, y_pred)
rmse_test =mean_squared_error(ytest,y_pred,squared=False) 
print(rmse_test) 
#0.3

#####
#Current Names
dfCurrentNames = dfCurrent.iloc[:, 1]
dfCurrentPredict = dfCurrent[['BLK', 'STL', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
dfCurrent.head(10)
#Random Forest

rfList = sorted(zip(dfCurrentNames, rfc.predict(dfCurrentPredict).tolist()), key=itemgetter(1), reverse=True)

x_rf = np.arange(len(rfList))

plt.style.use('fivethirtyeight')
rf, ax = plt.subplots()

ax.bar(x_rf, [row[1] for row in rfList], width=0.7, edgecolor='white', color='palegreen', linewidth=4, label='Predicted')

for rect, label in zip(ax.patches, [row[0] for row in rfList]):
    height = rect.get_height() + .02 if rect.get_x() > 6 else .03
    ax.text(rect.get_x() + rect.get_width() / 1.75, height, label, ha='center', va='bottom', rotation='vertical', color='black')

rf.suptitle("RF predicted MVP share", weight='bold', size=18, y=1.005)
ax.xaxis.set_visible(False)
ax.set_ylabel("Vote Share")
rf.text(x=-0.02, y=0.03, s='_' * 63, fontsize=14, color='grey', horizontalalignment='left')



