
# In[1]


# importing necessary libraries for further work
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


# ### Clean NA values and Creating Features

# In[2]:


# cleaning dataset before analysis
df = pd.read_csv('song_data.csv', skip_blank_lines = True)
df.dropna(inplace = True)


# In[3]:


#creating df with our CSV file of 19K songs
df.head()


# In[4]:


#changing column milliseconds to seconds
df = df.rename(columns = {'song_duration_ms' : 'seconds'})
df['seconds'] = df['seconds'].apply(lambda x: round(x / 1000))
df['seconds']


# In[5]:


# adding in ratings of popularity to group into 3 different buckets
# >= 70: popular, <70 and >=50: moderate: <50: unpopular
def popularity_rating(number):
    
    if number >= 70:
        return 'very'
    elif number >= 50: 
        return 'moderatly'
    else:
        return 'unpopular'


# In[6]:


df['rating'] = df['song_popularity'].apply(popularity_rating)


# ### Extracting Features and Target Variable(s)

# In[7]:


#num of ratings in each category
df['rating'].value_counts()


# In[8]:


# TARGET VARIABLE
target = df['rating']
target


# In[9]:


# ALL FEATURE VARIABLES
features = df.iloc[:, 2 : -1]
features.head()


# ### Preprocessing

# In[10]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scalers = [MinMaxScaler(), StandardScaler()]

for i in range(len(scalers)):

    #create the scaler
    scaler = scalers[i]

    #fit the scaler to the training data(features only)
    scaler.fit(features) 

    if i == 0:
        standard_scaled = scaler.transform(features)
    elif i == 1:
        minmax_scaled = scaler.transform(features)


# In[11]:


# New dataframe for all the scaled features
features_standard_scaled = pd.DataFrame(standard_scaled, columns = features.columns)
features_standard_scaled.head()


# In[12]:


# New dataframe for all the scaled features
features_minmax_scaled = pd.DataFrame(minmax_scaled, columns = features.columns)
features_minmax_scaled.head()


# In[13]:


from sklearn.model_selection import train_test_split

#split data into training and testing sets
def split_data(features, target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = 3000)
    
    return X_train, X_test, y_train, y_test


# In[14]:


# Creates training and testing data 
X_train, X_test, y_train, y_test = split_data(features, target)


# ### Feature Selection

# In[15]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# Uses a Decision Tree Classifier to eliminte features that have little to no effect on the target variable
select = RFE(DecisionTreeClassifier(random_state = 3000), n_features_to_select = 6)


# In[16]:


#fit the RFE selector to the training data
select.fit(X_train, y_train)

#transform training and testing sets so only the selected features are retained
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)


# In[17]:


# determines which of the features were deemed as most important
rfe_columns = []
for i in range(len(select.get_support())):
    if select.get_support()[i] == True:
        rfe_columns.append(features.columns[i])


# In[18]:


# Six chosen features
rfe_columns


# ### Dataframes with the Selected Features

# In[19]:


# Dictionary with features both scaled and unscaled to evaluate the performance
all_data = {'UNSCALED FEATURES' : features, 'STANDARD SCALER' : features_standard_scaled,
            'MINMAX SCALER' : features_minmax_scaled}


# In[20]:


# Removes the unused features from the dataframe
features = features[rfe_columns]
features_standard_scaled = features_standard_scaled[rfe_columns]
features_minmax_scaled = features_minmax_scaled[rfe_columns]


# In[21]:


#creating sample dataframe with samller set of selected features to visualize data
sample_df = df.sample(frac = 0.05, random_state = 3000)


# In[22]:


sample_df


# ### 3.2. Data Exploration


# This correlation chart gave us an easy way to show the correlations 
# between all the variables. It is color coded by relationship of each 
# correlation. What's interesting is that there are very weak positive 
# and negative correlations between most of the variables. However, it's 
# interesting to note that loudness and energy have a strong positive 
# correlation at 0.74.

# In[23]:


# correlation chart with all variables - features + target
corr = sample_df.corr()
corr.style.background_gradient(cmap="YlGn")


# The following scatterplots gave more insight into the relationships among 
# the features in this dataset. Some features such as Energy, Loudness and 
# Acousticness and Loudness seem to have linear/exponential relationships. 
# However, the vast mojority of variables do not seem to have strong linear 
# correlations with one another. In addition, the scatter matrix makes it 
# abundntly clear that we the data points across the different categories 
# tend to behave in very similar fashions.<center/>
    
# ** https://imgur.com/aCe1PxT

# In[24]:


#scatter matrix of 6 features against each other 
# visualization of correlations from above 

import plotly.express as px
fig = px.scatter_matrix(sample_df, dimensions = rfe_columns, color = 'rating', hover_name = 'song_name')
fig.show()




# The following histograms and barplots helped us understand the 
# overall distribution for each of the features in our analysis. The most 
# notable inforamtion that we acquired from this visualization include that 
# Acousticness and Liveliness have a significant positve skew while energy, 
# loudness, and audio valence have a negative skew. This does indicate that 
# a majority of the dataset share similar values for some of the features 
# which is indicative of having lower accuracies for our machine learning 
# models.

# * Graph images:
#     * https://imgur.com/GU4uhs1
#     * https://imgur.com/DxOhX4T
#     * https://imgur.com/k5UWvxD
#     * https://imgur.com/vHnp7Kj
#     * https://imgur.com/XusbS7X
#     * https://imgur.com/ISidS1x
#     

# In[25]:


#plotting frequencies against variables
 
def plot_bar_graph(variable):
    plot= df[variable]
    plot_value= plot.value_counts()
    plt.figure(figsize=(10,5))
    plt.bar(plot_value.index, plot_value, color="g")
    plt.ylabel("Frequency")
    plt.xlabel(variable)
    plt.show()

bar_plots = rfe_columns
for i in bar_plots:
    plot_bar_graph(i)


# 



# The boxplots allowed for an enhanced understanding about the spread of 
# the data for the individual features and how they compare to the other 
# target variables. The data points for each target variable- very, moderately, 
# and unpopular had fairly similar distributions for each of the target 
# variables included in the analysis.

# * Graph images:
#     * https://imgur.com/UbkvXJC
#     * https://imgur.com/TH01Rqv
#     * https://imgur.com/JxkXk19
#     * https://imgur.com/thTS5Bs
#     * https://imgur.com/4yBL4Np
#     * https://imgur.com/37BgfSL
#   

# In[26]:


box_plots = rfe_columns
def popularity_var(variable):
    graph = df.boxplot(column=[variable], by=['rating'])
    plt.title(variable)
    plt.figure(figsize=(7,7))
    plt.show()
    plt.savefig("box_plot"+variable)
for i in box_plots:
    popularity_var(i)


# In[27]:


df_numbers = df.iloc[:, 1:]


# In[28]:


df_numbers.groupby('rating').agg(['count', 'mean', 'std', 'sem'])




# We made the following scatter plots to understand how each of the features 
# affected the popularity rating of the song. We uncovered that there were no 
# significant linear or logistic relationships between the two variables. 
# In fact, the data set seems to show zero to low correlation between 
# popularity and the variables plottted.

# * Graph images:
#     * https://imgur.com/rdpp04f
#     * https://imgur.com/whzqkWh
#     * https://imgur.com/ZFzrbBj
#     * https://imgur.com/TO4PeYh
#     * https://imgur.com/3Rrabww
#     * https://imgur.com/bWDUxk0
#   

# In[29]:


for name in rfe_columns:
    fig = px.scatter(df, x = name, y = 'song_popularity', hover_name = 'song_name',
                     template = 'ggplot2', color = 'song_popularity',
                     color_continuous_scale = 'algae')

    fig.update_xaxes(title_font = {'size' : 18, 'family' : 'Courier', 'color' : 'gray'},
                     tickfont = {'size' : 16, 'family' : 'Courier', 'color' : 'gray'})

    fig.update_yaxes(title_font = {'size' : 18, 'family' : 'Courier', 'color' : 'gray'},
                     tickfont = {'size' : 16, 'family' : 'Courier', 'color' : 'gray'})
    fig.show()


# ### 3.3. Model Construction

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Dictionary of untuned classification algorithms
estimators = {
    'k-nearest neighbors' : KNeighborsClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Gaussian Naive Bayes' : GaussianNB(priors = None, var_smoothing = 1e-09),
    'Support Vector Machine' : LinearSVC(),
}


# In[31]:


#percentage-split method
def classifiers_percentage_split():
    
    for estimator_name, estimator_object in estimators.items():
        
        model = estimator_object
        model.fit(X_train, y_train)
        training_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(estimator_name)
        print('\t' + f'Prediction accuracy on the training data: {training_accuracy: .2%}')
        print('\t' + f'Prediction accuracy on the test data: {test_accuracy: .2%}\n')


# In[32]:


# tests the performance for each of the following datasets: untuned, minmax scaler, and standard scaler
for scaler_name, scaler_data in all_data.items():
    
    print(scaler_name, '\n-----------------\n')
    
    X_train, X_test, y_train, y_test = split_data(scaler_data, target)
    
    classifiers_percentage_split()


# ### 3.4. Model Evaluation

# In[33]:


from sklearn.metrics import classification_report

for estimator_name, estimator_object in estimators.items():
    class_report = classification_report(y_true = y_test, y_pred = estimator_object.predict(X = X_test))
    print(estimator_name + '\n-----------------------\n' + class_report + '\n\n')


# In[34]:


# Standard Scaler had the best overall performance
X_train, X_test, y_train, y_test = split_data(features_standard_scaled, target)


# <center> <h3>Algorithm: kNN Classifier </h3></center>

# In[35]:


param_grid_knn = {'n_neighbors' : [3, 5, 7, 10], 'weights' : ['uniform', 'distance']}


# In[36]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv = 5)

#fit the grid search object on the training data (CV will be performed on this)
grid_search.fit(X = X_train, y = y_train)

#result of grid search
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

#the performance of the best found parameters on the test set
#this is what you report for the evaluation of your model
print("Test set score: ", grid_search.score(X_test, y_test))


# <center> <h3>Algorithm: Decision Tree Classifier </h3></center>

# In[37]:


param_grid_dtc = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'],
                  'max_features' : [None, 'auto', 'sqrt', 'log2'], 'min_samples_split' : [2, 3, 4]}


# In[38]:


from sklearn.tree import DecisionTreeClassifier

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid_dtc, cv = 5)

#fit the grid search object on the training data (CV will be performed on this)
grid_search.fit(X = X_train, y = y_train)

#result of grid search
print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

#the performance of the best found parameters on the test set
#this is what you report for the evaluation of your model
print("Test set score: ", grid_search.score(X_test, y_test))


# In[39]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# the estimators dictionary is updated with the two tuned algorithmns that performed the best
estimators = {
    'k-nearest neighbors' : KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
                                                 metric_params = None, n_jobs = None, n_neighbors = 10, p = 2,
                                                 weights = 'distance'),
    
    'Decision Tree' : DecisionTreeClassifier(class_weight = None, criterion = 'entropy', max_depth = None,
                                             max_features = 'log2', max_leaf_nodes = None,
                                             min_impurity_decrease = 0.0, min_impurity_split = None,
                                             min_samples_leaf = 1, min_samples_split = 2,
                                             min_weight_fraction_leaf = 0.0,
                                             random_state = None, splitter = 'random')
}



# In[40]:


classifiers_percentage_split()


# In[41]:


estimators


# In[42]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def classifiers_cross_validation():

    for estimator_name, estimator_object in estimators.items():

        kfold = KFold(n_splits = 10, random_state = 3000, shuffle = True)
        scores = cross_val_score(estimator = estimator_object, X = features, y = target, cv = kfold)
        print(estimator_name + ':\n\t' + f'Mean Accuracy = {scores.mean():.2%}, ',
              f'Standard Deviation = {scores.std():.2%}' + '\n')


# In[43]:


classifiers_cross_validation()


# In[44]:


from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
                           metric_params = None, n_jobs = None, n_neighbors = 10, p = 2,
                           weights = 'distance')

knn.fit(X = X_train, y = y_train)

predicted = knn.predict(X = X_test)

expected = y_test


confusion = confusion_matrix(y_true = expected, y_pred = predicted)


# In[45]:


confusion_df = pd.DataFrame(confusion, index = ['moderatly', 'unpopular', 'very'],
                            columns = ['moderatly', 'unpopular', 'very'])
confusion_df




# In[46]:


for estimator_name, estimator_object in estimators.items():
    class_report = classification_report(y_true = y_test, y_pred = estimator_object.predict(X = X_test))
    print(estimator_name + '\n-----------------------\n' + class_report + '\n\n')


