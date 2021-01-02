#!/usr/bin/env python
# coding: utf-8

# <center> <h2> DS 3000 - Summer 2020</h2> </center>
# <center> <h3> DS Report </h3> </center>

# 
# <center> <h3> Forecasting the Popularity of Songs on Spotify</h3> </center>
# <center><h4>Aneesha Sreerama and Riya Chandra</h4></center>
# 

# <hr style="height:2px; border:none; color:black; background-color:black;">

# #### Executive Summary:
# 
# Spotify is the Swedish music streaming service that has become one of the most popular platforms in the world- especially for college students. As a group, we have always been interested in how songs become popular, and specifically how they get classified into popular/recommended playlists. Are there trends or is it completely random? Our goal for this project is to determine if relationships exist between variables such as danceability, audio valence, etc. from Spotify's data and observe their correlation with popularity. Using this information, we aim to create a predictability model for Spotify songs. 
# 
# Through a dataset published on Kaggle, we obtained a file of ~19,000 cells with a variety of Spotify features as well as song popularity, our target variable. While song popularity was given to us as a continuous variable, we decided to turn it into a multi-classification problem by grouping the song popularity variable into three categories: popular, moderate, unpopular. Using pandas, we transformed the CSV file into a dataframe to start cleaning the dataset. We normalized our datasets using MinMaxScaler and StandardScaler and used RFE for feature selection. Moreover, we conducted an exploratory analysis by creating data visualizations such as scatterplots, bar plots, histograms, box plots, and a correlation chart to understand relationships between varaibles. Then we constructed our model using k-Nearest Neighbor, Decision Tree Classifier, Gaussian Naive-Bayes, and SVM. When optimizing the model, we performed Grid-Search on Decision Tree and K-Nearest Neighbors as they proved to have the highest accuracy. Through the use of cross-validation we obtained our final results. 

# <hr style="height:2px; border:none; color:black; background-color:black;">

# ## Outline
# 1. <a href='#1'>INTRODUCTION</a>
# 2. <a href='#2'>METHOD</a>
# 3. <a href='#3'>RESULTS</a>
# 4. <a href='#4'>DISCUSSION</a>

# <a id="1"></a>
# <hr style="height:2px; border:none; color:black; background-color:black;">

# ## 1. INTRODUCTION

# **Problem Statement**
# 
# Our project addresses the following problem: What features and qualities are characteristic of a popular song? Characteristics of songs are measured in many different numerical values such as tempo, time in seconds, acousticness, etc. Spotify releases data that measure these said features for all the songs in their database. As Spotify is a primary music platform for college students around the world, we chose to investigate the trends and relationships between song features and how can we predict which songs will become popular moving forward. 
# 
# Our initial step is to explore the relationship between the features of the song that are provided in the dataset. Moreover, we plan to split the song popularity into three buckets: popular, moderate, unpopular and see if there are any are trends specific to the respective buckets. Using this information, we will create a predictability model based on selected feature variables. 
# 
# 
# **Significance of the Problem**
# 
# With competing digital music platforms like Spotify, Pandora, Apple Music, it can be important to use predictability techniques to predict popular songs in the future and group them into categories - for example, predicting which songs will be popular for the "New Music Friday" playlist on Spotify. Even artists and music agencies can use our findings to determine what kinds of songs become the most popular and tailor future music based on it. In addition to this, a prediction model can help with unsupervised algorithms like recommendation machines for specific users. 
# 
# 
# **Previous Work on the Topic** 
# 
# Unsurprisingly, there is a lot of previous information on Spotify due to the amount and specificity of data that they release to us. Spotify has a robust API and a lot of datasets published with all their songs (like the one we used) and Top 50 songs of each year. Similar to our findings, using all kinds of large and small datasets, many people found that there were not many relationships between the feature variables, except energy and loudness which was consistent through many findings including ours. In previous projects, people used linear and logistic regression to create a predictability model. In many linear regression models, it was found that daceability could create a higher popularity score while energy and instrumentalness could lower the score. When conducting our project, we decided to take a different approach and make it a multi-classification model as we thought it would add more complexity and accurracy with splitting the data vs. using a continuous variable as our target.
# 
# **Research Questions**
# * Given the aforementioned problem and its importance, we set out to tackle the following questions:
#     * Are there specific features that directly correlate to song popularity? 
#     * Which feature variables are more indicative of projected song popularity?
#     * Will the accuracy of our predictions increase when we recursively eliminate features? Will certain algorithms perform better when there are more features in evaluation? 
#     * A popularity of a song is dependent on a vast number of features that can be very difficult. Given the features that we do take into account, will knn work significantly better than a Decision Tree Classifier or a Support Vector Machine Classifier?
# 
# **Hypotheses:**
# 
# * **Hypothesis about features relating to popularity:**
#      * Null: Neither danceability, energy, or instrumentalness is associated with a high popularity rating.
#      * Alternative: Either danceability, energy, or instrumentalness is associated with a high popularity rating.
#      
# * **Hypothesis about features relating to one another:**    
#      * Null: There is no strong linear correlation between loudness and energy.
#      * Alternative: There is a strong linear correlation between loudness and energy.
#        
# * **Hypothesis about machine learning algorithms**:
#      * Null: The k-Nearest Neighbor Algorithm is the best predictor for classifying song predictability
#      * Alternative: The k-Nearest Neighbor Algorithm is not the best predictor for classifying song predictability
# 
# * **Hypothesis about scaling the algorithms**:
#      * Null: Scaling the features with Standard Scaler will result in the same increase in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
#      * Alternative: Scaling the features with Standard Scaler will not result in the same increase (> 1% difference) in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
#        
#        
# #### References
# * https://blog.salesflare.com/spotify-iconic-product
# * https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md
# * https://techxplore.com/news/2019-09-spotify-songs.html
# * https://www.digitaltrends.com/music/apple-music-vs-spotify/
# * https://medium.com/@RyanJosephHill/why-spotify-is-one-of-my-favorite-products-93fa4dff850a

# <a id="2"></a>
# <hr style="height:2px; border:none; color:black; background-color:black;">

# ## 2. METHOD

# ### 2.1. Data Acquisition
# 
# * We found a dataset on Kaggle, which has 19,000 songs from Spotify along with features to describe each song: song name, song_duration, acousticness, danceability, energy, instrumental, key, liveness, loudness, audio_mode, speechiness, tempo, time_signature, and audio_valence. In addition to this, each song has a popularity ranking that ranges from 1-100. Our project will focus on looking at custom groups we create - Popular, Moderate, Unpopular to identify trends on how songs perform
# https://www.kaggle.com/edalrami/19000-spotify-songs
# 

# ### 2.2. Variables
# * Independent variables from our hypothesis will include danceability, energy, instrumentalness, and kNN regression algorithm with dependent variables as popularity and accuracy.
# * Feature Variables: 
#     * song name: name of song
#     * song_duration: duration of track in milliseconds
#     * acousticness: confidence measure from 0.0 to 1.0 of whether the track is acoustic (1.0 represnts high confidence that track is acoustic
#     * danceability: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and regularity - 0.0 is least daceable and 1.0 is most danceable
#     * energy: measure from 0.0 to 1.0 that presents a perceptual measure of intensity and activity 
#     * instrumentalnessL predicts whether a track contains no vocals "ooh" and "aah" sounds are treated as instrumental in this context - closer to 1.0, greater likelihood that the track contains no vocal content
#     * key: estimated overall key of track - if no key was detected, value is -1
#     * liveness: detects presence of an audience in the recording - higher liveness values represent an increased probability that the track was performed live, value of above 0.8 provides strong likelihood that the track is live
#     * loudness: overall loudness of track in decibels (dB) - loudness values are averaged across entire track and are useful for comparing relative loudness of tracks - quality of a sound that is primary psychological correlate of physical strength
#     * audio_mode: mode indiicates modality (major or minor) of a track, type of scale from which its melodic content is derived
#     * speechiness: detects presence of spoken words in a track - more exclusively speech-like the recording, closer to 1.0 the attribute value 
#     * tempo: overall estimated tempo of a track in beats per minute, tempo in music is the speed or pace of a given piece and derives directly from the average beat duration
#     * time_signature: estimated ooverall time signature of a track - time signature is a notational convention to specify how many beats are in each bar (or measure)
#     * audio_valence: measure from 0.01-1.0 describing musical positiveness conveyed by a track - tracks with high valence sound more positive while tracks with low valence sound more negative
# * Target Variable: popularity rating (Popular, Moderately, Unpopular)
# 

# ### 2.3. Data Analysis
# * Our project is tackles a multi-class classification problem. The first step we will take is cleaning the data and classifying them into three groups - not popular, somewhat popular, and very popular based on the song_popularity numeric feature. We will also look at the various correlations between the features to determine any patterns and relationships. In addition, we will use iterative feature engineering to determine the importance of features. We will also use min/max and standard scalar to normalize the features.
# 
# * Our target variable is popularity and we have separated popularity out into three classes making it a categorical variable. We are planning to use KNN, SVM, Naive Bayes, and Decision Tree Classifiers along with GridSearch. We are not sure which algorithms will perform the best, so we plan to give them all a try and compare the accuracies.
# 
# **Predictive Model**:
#  We are going to be predicting popularity of Spotify songs using a testing & training dataset
# 
# **A Supervised Learning Problem**: This is a supervised learning problem, categorized into classification. We will be dividing popularity into three categories - popular, moderate, unpopular making it a categorical variable. 
# 
# **Machine Learning Algorithms to be Applied**:
# 
# Based on research on predictability models, we incorporated the following classification algorithms as they can categorize the data points into the popularity rating groups created.
# 
# 
# * K-Neighbors Classifier
# * Decision Tree Classifier
# * Naive Bayes
# * Support Vector Machines
# 

# <a id="3"></a>
# <hr style="height:2px; border:none; color:black; background-color:black;">

# ## 3. RESULTS

# ### 3.1. Data Wrangling

# In[1]:


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
# 
# 

# <center><h7>Description of Correlation Chart<h7/><center> 

# <center>This correlation chart gave us an easy way to show the correlations between all the variables. It is color coded by relationship of each correlation. What's interesting is that there are very weak positive and negative correlations between most of the variables. However, it's interesting to note that loudness and energy have a strong positive correlation at 0.74. <center/>

# In[23]:


# correlation chart with all variables - features + target
corr = sample_df.corr()
corr.style.background_gradient(cmap="YlGn")


# <center><h7>Description of Scatter Matrix<h7/><center> 

# <center>The following scatterplots gave more insight into the relationships among the features in this dataset. Some features such as Energy and Loudness and Acousticness and Loudness seem to have linear/exponential relationships. However, the vast mojority of variables do not seem to have strong linear correlations with one another. In addition, the scatter matrix makes it abundntly clear that we the data points across the different categories tend to behave in very similar fashions.<center/>
#     
# ** https://imgur.com/aCe1PxT

# In[24]:


#scatter matrix of 6 features against each other 
# visualization of correlations from above 

import plotly.express as px
fig = px.scatter_matrix(sample_df, dimensions = rfe_columns, color = 'rating', hover_name = 'song_name')
fig.show()


# <center><h7>Description of Barplots<h7/><center> 

# <center>The following histograms and barplots helped us understand the overall distribution for each of the features in our analysis. The most notable inforamtion that we acquired from this visualization include that Acousticness and Liveliness have a significant positve skew while energy, loudness, and audio valence have a negative skew. This does indicate that a majority of teh dataset share similar values for some of the features which is indicative of having lower accuracies for our machine learning models.<center/>

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

# <center><h7>Description of Box Plots<h7/><center> 

# <center>The boxplots allowed for an enhanced understanding about the spread of the data for the individual features and how they compare to the other target variables. The data points for each target variable- very, moderately, and unpopular had fairly similar distributions for each of the target variables included in the analysis.<center/>

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


# <center><h7>Description of Scatter Plots<h7/><center> 

# <center>We made the following scatter plots to understand how each of the features affected the popularity rating of the song. We uncovered that there were no significant linear or logistic relationships between the two variables. In fact, the data set seems to show zero to low correlation between popularity and the variables plottted.<center> 
# 
# 

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


# **Model Evaluation:**
# 
# * **k-Nearest Neighbors Classifier:**
#      * Test Set Accuracy: ~ 49%
#      * Evaluation: The k-Nearest Neighbors Classifier had an approximate 49.39% accuracy on the test set, and a 66.66% accuracy on the training data. This is indicative of mild overfitting of the training data. There was a significant difference between the two values. This algorithm has a lot of potential for improvement if we can tune parameters such as the k value and the metric for determining which points are closer. The F1 scores for unpopular, moderatley, and very were 0.47, 0.50, 0.52 respectively. This tells us that both the precision and recall were fairly low. Tuning will help minimize the chance of not detecting a positive case and minimize the number of false positives.
#      
# * **Decision Tree Classifier:**    
#      * Test Set Accuracy: ~ 58%
#      * Evaluation: The Decision Tree Classifier had an approximate 58.99% accuracy on the test set, and a 99.32% accuracy in the training data. This is indicative of massive overfitting of the training data as there was a significant difference between the two values. This algorithm has a lot of potential for improvement if we can tune parameters such as the splitter and criterion. We need to loosen the fit of the data by optimizing these values. The F1 scores for unpopular, moderatley, and very were 0.56, 0.55, and 0.69 respectively. This tells us that both the precision and recall are quite low but the F1 score for the 'Very Popular' category was higher than that of the k-Nearest Neighbor Classifier. Tuning will help minimize the chance of not detecting a positive case and minimize the number of false positives.
#        
# * **Naive Bayes**:
#      * Test Set Accuracy: ~ 35%
#      * Evaluation: The Naive Bayes Classifier performed with a very poor accuracy and had low F1 scores across the board. The Training and Test Set accuracies were approximatly the same indicating that there was no significant underfitting or overfitting of the model. Tuning this algorithm would not provide any benefits for this reason.
# 
# * **Support Vector Machine**:
#      * Test Set Accuracy: ~ 41%
#      * Evaluation: The Support Vector Machine Classifier performed with a very poor accuracy and had low F1 scores across the board. The Training and Test Set accuracies were approximatly the same indicating that there was no significant underfitting or overfitting of the model. Tuning this algorithm would not provide any benefits for this reason.
# 
# 
# **Hypotheses Evaluation:**
# 
# * **Hypothesis about features relating to popularity:**
#      * Null: Neither danceability, energy, or instrumentalness is associated with a high popularity rating.
#      * Alternative: Either danceability, energy, or instrumentalness is associated with a high popularity rating.
#      * Evaluation: According to the correlation matrix that we constructed, danceability, energy, and instrumentalness have the pearson correaltion coefficeints of 0.09467, 0.00037, and -0.161584 respectively when plotted against popularity rating. This is indicative of very low correlation between each of the variables and popularity. In addition, the scatter plot showed that there was no observable exponential or logistic relationship between each of the varibles and popularity. Based on the data, we would fail to reject the null hypothesis.
#      
# * **Hypothesis about features relating to one another:**    
#      * Null: There is no strong linear correlation between loudness and energy.
#      * Alternative: There is a strong linear correlation between loudness and energy.
#      * Evaluation: According to the correlation matrix that we constructed, loudness and energy had a correaltion coefficient of 0.74368. This is indicative of a strong linear relationship between the two features. The notion that loudness and energy are generally positively associated with one another is supported by this data. Although, the scatter matrix does suggest that the two varibles can be better modeled with a logistic curve, we can reject the null hypothesis. There is a strong linear correaltion between loudness and energy.
#        
# * **Hypothesis about machine learning algorithms**:
#      * Null: The k-Nearest Neighbor Algorithm is the best predictor for classifying song predictability
#      * Alternative: The k-Nearest Neighbor Algorithm is not the best predictor for classifying song predictability
#      * Evaluation: According to the class report, the Decision Tree Classifier has ann accuracy of 59% whereas the k-Nearest Neighbor Classifier only has an accuracy of 49%. In addition, the reported statistics indicate that the Decsion Tree Classifier is massively overfitted as the training set accuracy is 99.32% whereas the testing set accuracy is 58.38%. Similarly the k-Nearest Neighbor Algorithm is quite overfitted as the the training set accuracy is 66.66% whereas the testing set accuracy is 49.39%. Although both algorithms have potential for significant improvement, at this stage the Decision Tree Classifier is the better predictor for classifying song predictability. Therefore we can reject the Null Hypothesis.
# 
# * **Hypothesis about scaling the algorithms**:
#      * Null: Scaling the features with Standard Scaler will result in the same increase in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
#      * Alternative: Scaling the features with Standard Scaler will not result in the same increase (> 1% difference) in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
#      * Evaluation: The reported statistics indicate that after using Stanrd Scaler, the performance of the Support Vector Machine Classifier increased from 37.93% to 41.07% on the testing data. On the other hand, the performance of the K-Nearest Neighbor Classifier increased from 48.40% to 49.39% on the testing data. The effect of scaling had a larger impact for the SVM Classifier. Therefore, we can reject the null hypothesis.
#        

# ### 3.5. Model Optimization

# **Choosing the Best Predicitve Models**
# 
# * **k-Nearest Neighbors Classifier:**
#      * The training and testing accuracies are indicative of mild overfitting of the training data. Since, there was a significant difference between the two values, this algorithm has a lot of potential for improvement. We can tune the parameters such as the number of nearest neighbors and the metric for determining which points are closer. We accomplished this task underfitting the model through the use of Grid Search.
#      
#      
# * **Decision Tree Classifier:**
#     * The training and testing accuracies are indicative of massive overfitting of the training data. Since, there was a very significant difference between the two values, this algorithm has a lot of potential for improvement. We can tune the following parameters: criterion, splitter, max_feaures, and minimum samples to split in order to improve both the precision and accuracy of the algorithm. We accomplished the task of underfitting the model through the use of Grid Search.
# 

# <center> <h2>Grid Search</h2></center>

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


# ### 3.6. Model Testing
# 

# <h4>Technique: Using Percentage Split </h4>

# In[40]:


classifiers_percentage_split()


# <h4>Technique: Using Cross Validation </h4>

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


# <h4>Understanding the Performance with Confusion Matrices </h4>

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


# <center>The following confusion matrix gives more insight into how the different target values were categorized. This tells us that the frequency at which moderate and unpopular songs were mistaken for each other was very significant. The distinction between 'Very Popular' and the other two categories was more clear.<center/>

# In[46]:


for estimator_name, estimator_object in estimators.items():
    class_report = classification_report(y_true = y_test, y_pred = estimator_object.predict(X = X_test))
    print(estimator_name + '\n-----------------------\n' + class_report + '\n\n')


# <a id="4"></a>
# <hr style="height:2px; border:none; color:black; background-color:black;">

# ## 4. DISCUSSION
# 

# ### Summary of Analysis
# 
# In order to analyze our data, we began by splitting our data into the feature and target variables. We scaled our data uning Standard and Minmax Scaler in order to compare the accuracies of the preliminary models with the data. We tested the four main classification algorithms: k-Nearest Neighbor Classifer, Decision Tree Classifier, Support Vector Machine, and Gaussain Naive Bayes. The accuracy of the test and training sets indicated that the k-Nearest Neighbor Classifer and the Decision Tree Classifier were thoroughly overfitting the training data. We proceeded to tune these algorthms in order to loosen the fit on the training data. Through the use of Grid Search, we observed that the accuracy of the testing set data improved significantly on the test data set for the k-Nearest Neighbor. This resulted in a a very similar performance for the k-Nearest Neighbor Classifier and Decision Tree Classifier at around a 60% accuracy on the test data set. This analysis resulted in prediction accuracies that were averagely successful.
# 
# ### Interpretation of Findings
# 
# ***Algorithms Compared***
# 
# We compared k-Nearest Neighbor Classifier, Decision Tree Classifier, Support Vector Machine, and Gaussian Naive Bayes algorithms
# 
# ***Algorithms with Best Performance***
# 
# The k-Nearest Neighbor Classifier outperformed the Decision Tree Classifier with the best F1 scores and Accuracies on the testing sets:
# 
#     k-Nearest Neighbor Classifier
#     -----------------------------
#     * Accuracy: ~ 57%
#     * F1 Score for 'Very Popular': 0.68
#     * F1 Score for 'Moderately Popular': 0.54
#     * F1 Score for 'UnPopular': 0.54
#     
#     Decision Tree Classifier
#     ------------------------
#     * Accuracy: ~ 57%
#     * F1 Score for 'Very Popular': 0.68
#     * F1 Score for 'Moderately Popular': 0.53
#     * F1 Score for 'UnPopular': 0.52
#     
# ***Evaluation after Optimization***
# 
# After tuning and optimization, we uncovered a huge improvement in the performace of the k-Nearest Neighbor Classifier. The accuracy of the k-Nearest Neighbor Algorithm increased by a nearly whopping 10%. On the other hand, the Dscision Tree Classifier had a mild improvements in performance. However, for each of the algorithms, the tuned model continued to massively overfitted the training dataset whilst improving the accuracy on the test set data.
# 
# ***Algorithms for Use in Predictive Model***
# 
# We determined that the k-Nearest Neighbor Classifier and Decision Tree had the highest accuracy so we utilized them to perform our predictive model. Based on our results, we determined that the features in our dataset do not give us a complete understanding of the entire music industry as we would have liked it to. Given this reasoning, it seems fair to assume that either of these classifiers can be used to yield a high accuracy rate on the given dataset.
#    
# 
# ### Our Original Research Questions
# 
# ***Are there specific features that directly correlate to song popularity?***
# 
# No, none of the feature variables had a strong positive or negative correlation to song popularity.
# 
# ***Which feature variables are more indicative of projected song popularity?***
# 
# Using RFE, we conducted a feature selection and found that acousticness, danceability, energy, liveness, and loudness were the most indicative features of song popularity. 
# 
# ***Will the accuracy of our predictions increase when we recursively eliminate features? Will certain algorithms perform better when there are more features in evaluation?***
# 
# We tried testing it on 4, 5, 6, and 7 features and found that 6 features resulted in algorithms that performed better. 
# 
# ***A popularity of a song is dependent on a vast number of features that can be very difficult. Given the features that we do take into account, will knn work significantly better than a Decision Tree Classifier or a Support Vector Machine Classifier?***
# 
# After performing cross-validation, we found that the accuracy of k-Nearest Neighbors was slightly higher than the Decision Tree Classifier. 
# 
# 
# ***Reflection on our Findings***
# 
# Going back to our original project topic, we wanted to determine whether there were any relationships between our feature variables and song popularity. We were surprised to find that there were no strong correlations. 
# 
# However, when conducting our model, we actually found that when using six feature variables, there was a fairly high accuracy of 60% using the k-Nearest Neighbor algorithm. 
# 
# 
# ### Conclusion
# 
# One way to have more accurate results would be to increase the number of feature variables. We only used feature variables that were given to us by Spotify, but if we added more variables such as social media response time to songs or number of tweets about a specific song, we may have found a higher accuracy for our predictive model.
# 
# In addition to this, when looking up past research, we found many predictability models that used regression instead of classification. We would be curious to see if we kept song_popularity as a continuous variable, instead of classifying it into 3 buckets, would accuracy be higher? This is something we’d love to conduct in the future.
