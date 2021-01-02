# Spotify-Analysis

#### Executive Summary:

Spotify is the Swedish music streaming service that has become one of the most popular platforms in the world- especially for college students. As a group, we have always been interested in how songs become popular, and specifically how they get classified into popular/recommended playlists. Are there trends or is it completely random? Our goal for this project is to determine if relationships exist between variables such as danceability, audio valence, etc. from Spotify's data and observe their correlation with popularity. Using this information, we aim to create a predictability model for Spotify songs. 

Through a dataset published on Kaggle, we obtained a file of ~19,000 cells with a variety of Spotify features as well as song popularity, our target variable. While song popularity was given to us as a continuous variable, we decided to turn it into a multi-classification problem by grouping the song popularity variable into three categories: popular, moderate, unpopular. Using pandas, we transformed the CSV file into a dataframe to start cleaning the dataset. We normalized our datasets using MinMaxScaler and StandardScaler and used RFE for feature selection. Moreover, we conducted an exploratory analysis by creating data visualizations such as scatterplots, bar plots, histograms, box plots, and a correlation chart to understand relationships between varaibles. Then we constructed our model using k-Nearest Neighbor, Decision Tree Classifier, Gaussian Naive-Bayes, and SVM. When optimizing the model, we performed Grid-Search on Decision Tree and K-Nearest Neighbors as they proved to have the highest accuracy. Through the use of cross-validation we obtained our final results. 

**Problem Statement**

Our project addresses the following problem: What features and qualities are characteristic of a popular song? Characteristics of songs are measured in many different numerical values such as tempo, time in seconds, acousticness, etc. Spotify releases data that measure these said features for all the songs in their database. As Spotify is a primary music platform for college students around the world, we chose to investigate the trends and relationships between song features and how can we predict which songs will become popular moving forward. 

Our initial step is to explore the relationship between the features of the song that are provided in the dataset. Moreover, we plan to split the song popularity into three buckets: popular, moderate, unpopular and see if there are any are trends specific to the respective buckets. Using this information, we will create a predictability model based on selected feature variables. 


**Significance of the Problem**

With competing digital music platforms like Spotify, Pandora, Apple Music, it can be important to use predictability techniques to predict popular songs in the future and group them into categories - for example, predicting which songs will be popular for the "New Music Friday" playlist on Spotify. Even artists and music agencies can use our findings to determine what kinds of songs become the most popular and tailor future music based on it. In addition to this, a prediction model can help with unsupervised algorithms like recommendation machines for specific users. 


**Previous Work on the Topic** 

Unsurprisingly, there is a lot of previous information on Spotify due to the amount and specificity of data that they release to us. Spotify has a robust API and a lot of datasets published with all their songs (like the one we used) and Top 50 songs of each year. Similar to our findings, using all kinds of large and small datasets, many people found that there were not many relationships between the feature variables, except energy and loudness which was consistent through many findings including ours. In previous projects, people used linear and logistic regression to create a predictability model. In many linear regression models, it was found that daceability could create a higher popularity score while energy and instrumentalness could lower the score. When conducting our project, we decided to take a different approach and make it a multi-classification model as we thought it would add more complexity and accurracy with splitting the data vs. using a continuous variable as our target.

**Research Questions**
* Given the aforementioned problem and its importance, we set out to tackle the following questions:
    * Are there specific features that directly correlate to song popularity? 
    * Which feature variables are more indicative of projected song popularity?
    * Will the accuracy of our predictions increase when we recursively eliminate features? Will certain algorithms perform better when there are more features in evaluation? 
    * A popularity of a song is dependent on a vast number of features that can be very difficult. Given the features that we do take into account, will knn work significantly better than a Decision Tree Classifier or a Support Vector Machine Classifier?

**Variables**
* Independent variables from our hypothesis will include danceability, energy, instrumentalness, and kNN regression algorithm with dependent variables as popularity and accuracy.
* Feature Variables: 
    * song name: name of song
    * song_duration: duration of track in milliseconds
    * acousticness: confidence measure from 0.0 to 1.0 of whether the track is acoustic (1.0 represnts high confidence that track is acoustic
    * danceability: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and regularity - 0.0 is least daceable and 1.0 is most danceable
    * energy: measure from 0.0 to 1.0 that presents a perceptual measure of intensity and activity 
    * instrumentalnessL predicts whether a track contains no vocals "ooh" and "aah" sounds are treated as instrumental in this context - closer to 1.0, greater likelihood that the track contains no vocal content
    * key: estimated overall key of track - if no key was detected, value is -1
    * liveness: detects presence of an audience in the recording - higher liveness values represent an increased probability that the track was performed live, value of above 0.8 provides strong likelihood that the track is live
    * loudness: overall loudness of track in decibels (dB) - loudness values are averaged across entire track and are useful for comparing relative loudness of tracks - quality of a sound that is primary psychological correlate of physical strength
    * audio_mode: mode indiicates modality (major or minor) of a track, type of scale from which its melodic content is derived
    * speechiness: detects presence of spoken words in a track - more exclusively speech-like the recording, closer to 1.0 the attribute value 
    * tempo: overall estimated tempo of a track in beats per minute, tempo in music is the speed or pace of a given piece and derives directly from the average beat duration
    * time_signature: estimated ooverall time signature of a track - time signature is a notational convention to specify how many beats are in each bar (or measure)
    * audio_valence: measure from 0.01-1.0 describing musical positiveness conveyed by a track - tracks with high valence sound more positive while tracks with low valence sound more negative
* Target Variable: popularity rating (Popular, Moderately, Unpopular)

### 2.3. Data Analysis
* Our project is tackles a multi-class classification problem. The first step we will take is cleaning the data and classifying them into three groups - not popular, somewhat popular, and very popular based on the song_popularity numeric feature. We will also look at the various correlations between the features to determine any patterns and relationships. In addition, we will use iterative feature engineering to determine the importance of features. We will also use min/max and standard scalar to normalize the features.

* Our target variable is popularity and we have separated popularity out into three classes making it a categorical variable. We are planning to use KNN, SVM, Naive Bayes, and Decision Tree Classifiers along with GridSearch. We are not sure which algorithms will perform the best, so we plan to give them all a try and compare the accuracies.

**Predictive Model**:
 We are going to be predicting popularity of Spotify songs using a testing & training dataset

**A Supervised Learning Problem**: This is a supervised learning problem, categorized into classification. We will be dividing popularity into three categories - popular, moderate, unpopular making it a categorical variable. 

**Machine Learning Algorithms to be Applied**:

Based on research on predictability models, we incorporated the following classification algorithms as they can categorize the data points into the popularity rating groups created.


* K-Neighbors Classifier
* Decision Tree Classifier
* Naive Bayes
* Support Vector Machines

**Hypotheses:**

* **Hypothesis about features relating to popularity:**
     * Null: Neither danceability, energy, or instrumentalness is associated with a high popularity rating.
     * Alternative: Either danceability, energy, or instrumentalness is associated with a high popularity rating.
     
* **Hypothesis about features relating to one another:**    
     * Null: There is no strong linear correlation between loudness and energy.
     * Alternative: There is a strong linear correlation between loudness and energy.
       
* **Hypothesis about machine learning algorithms**:
     * Null: The k-Nearest Neighbor Algorithm is the best predictor for classifying song predictability
     * Alternative: The k-Nearest Neighbor Algorithm is not the best predictor for classifying song predictability

* **Hypothesis about scaling the algorithms**:
     * Null: Scaling the features with Standard Scaler will result in the same increase in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
     * Alternative: Scaling the features with Standard Scaler will not result in the same increase (> 1% difference) in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
       

**Model Evaluation:**

* **k-Nearest Neighbors Classifier:**
     * Test Set Accuracy: ~ 49%
     * Evaluation: The k-Nearest Neighbors Classifier had an approximate 49.39% accuracy on the test set, and a 66.66% accuracy on the training data. This is indicative of mild overfitting of the training data. There was a significant difference between the two values. This algorithm has a lot of potential for improvement if we can tune parameters such as the k value and the metric for determining which points are closer. The F1 scores for unpopular, moderatley, and very were 0.47, 0.50, 0.52 respectively. This tells us that both the precision and recall were fairly low. Tuning will help minimize the chance of not detecting a positive case and minimize the number of false positives.
     
* **Decision Tree Classifier:**    
     * Test Set Accuracy: ~ 58%
     * Evaluation: The Decision Tree Classifier had an approximate 58.99% accuracy on the test set, and a 99.32% accuracy in the training data. This is indicative of massive overfitting of the training data as there was a significant difference between the two values. This algorithm has a lot of potential for improvement if we can tune parameters such as the splitter and criterion. We need to loosen the fit of the data by optimizing these values. The F1 scores for unpopular, moderatley, and very were 0.56, 0.55, and 0.69 respectively. This tells us that both the precision and recall are quite low but the F1 score for the 'Very Popular' category was higher than that of the k-Nearest Neighbor Classifier. Tuning will help minimize the chance of not detecting a positive case and minimize the number of false positives.
       
* **Naive Bayes**:
     * Test Set Accuracy: ~ 35%
     * Evaluation: The Naive Bayes Classifier performed with a very poor accuracy and had low F1 scores across the board. The Training and Test Set accuracies were approximatly the same indicating that there was no significant underfitting or overfitting of the model. Tuning this algorithm would not provide any benefits for this reason.

* **Support Vector Machine**:
     * Test Set Accuracy: ~ 41%
     * Evaluation: The Support Vector Machine Classifier performed with a very poor accuracy and had low F1 scores across the board. The Training and Test Set accuracies were approximatly the same indicating that there was no significant underfitting or overfitting of the model. Tuning this algorithm would not provide any benefits for this reason.


**Hypotheses Evaluation:**

* **Hypothesis about features relating to popularity:**
     * Null: Neither danceability, energy, or instrumentalness is associated with a high popularity rating.
     * Alternative: Either danceability, energy, or instrumentalness is associated with a high popularity rating.
     * Evaluation: According to the correlation matrix that we constructed, danceability, energy, and instrumentalness have the pearson correaltion coefficeints of 0.09467, 0.00037, and -0.161584 respectively when plotted against popularity rating. This is indicative of very low correlation between each of the variables and popularity. In addition, the scatter plot showed that there was no observable exponential or logistic relationship between each of the varibles and popularity. Based on the data, we would fail to reject the null hypothesis.
     
* **Hypothesis about features relating to one another:**    
     * Null: There is no strong linear correlation between loudness and energy.
     * Alternative: There is a strong linear correlation between loudness and energy.
     * Evaluation: According to the correlation matrix that we constructed, loudness and energy had a correaltion coefficient of 0.74368. This is indicative of a strong linear relationship between the two features. The notion that loudness and energy are generally positively associated with one another is supported by this data. Although, the scatter matrix does suggest that the two varibles can be better modeled with a logistic curve, we can reject the null hypothesis. There is a strong linear correaltion between loudness and energy.
       
* **Hypothesis about machine learning algorithms**:
     * Null: The k-Nearest Neighbor Algorithm is the best predictor for classifying song predictability
     * Alternative: The k-Nearest Neighbor Algorithm is not the best predictor for classifying song predictability
     * Evaluation: According to the class report, the Decision Tree Classifier has ann accuracy of 59% whereas the k-Nearest Neighbor Classifier only has an accuracy of 49%. In addition, the reported statistics indicate that the Decsion Tree Classifier is massively overfitted as the training set accuracy is 99.32% whereas the testing set accuracy is 58.38%. Similarly the k-Nearest Neighbor Algorithm is quite overfitted as the the training set accuracy is 66.66% whereas the testing set accuracy is 49.39%. Although both algorithms have potential for significant improvement, at this stage the Decision Tree Classifier is the better predictor for classifying song predictability. Therefore we can reject the Null Hypothesis.

* **Hypothesis about scaling the algorithms**:
     * Null: Scaling the features with Standard Scaler will result in the same increase in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
     * Alternative: Scaling the features with Standard Scaler will not result in the same increase (> 1% difference) in performance for the Support Vector Machine Classifier and the K-Nearest Neighbor Classifier
     * Evaluation: The reported statistics indicate that after using Stanrd Scaler, the performance of the Support Vector Machine Classifier increased from 37.93% to 41.07% on the testing data. On the other hand, the performance of the K-Nearest Neighbor Classifier increased from 48.40% to 49.39% on the testing data. The effect of scaling had a larger impact for the SVM Classifier. Therefore, we can reject the null hypothesis.
     
     
     
**Choosing the Best Predicitve Models**

* **k-Nearest Neighbors Classifier:**
     * The training and testing accuracies are indicative of mild overfitting of the training data. Since, there was a significant difference between the two values, this algorithm has a lot of potential for improvement. We can tune the parameters such as the number of nearest neighbors and the metric for determining which points are closer. We accomplished this task underfitting the model through the use of Grid Search.
     
     
* **Decision Tree Classifier:**
    * The training and testing accuracies are indicative of massive overfitting of the training data. Since, there was a very significant difference between the two values, this algorithm has a lot of potential for improvement. We can tune the following parameters: criterion, splitter, max_feaures, and minimum samples to split in order to improve both the precision and accuracy of the algorithm. We accomplished the task of underfitting the model through the use of Grid Search.


### Summary of Analysis

In order to analyze our data, we began by splitting our data into the feature and target variables. We scaled our data uning Standard and Minmax Scaler in order to compare the accuracies of the preliminary models with the data. We tested the four main classification algorithms: k-Nearest Neighbor Classifer, Decision Tree Classifier, Support Vector Machine, and Gaussain Naive Bayes. The accuracy of the test and training sets indicated that the k-Nearest Neighbor Classifer and the Decision Tree Classifier were thoroughly overfitting the training data. We proceeded to tune these algorthms in order to loosen the fit on the training data. Through the use of Grid Search, we observed that the accuracy of the testing set data improved significantly on the test data set for the k-Nearest Neighbor. This resulted in a a very similar performance for the k-Nearest Neighbor Classifier and Decision Tree Classifier at around a 60% accuracy on the test data set. This analysis resulted in prediction accuracies that were averagely successful.

### Interpretation of Findings

***Algorithms Compared***

We compared k-Nearest Neighbor Classifier, Decision Tree Classifier, Support Vector Machine, and Gaussian Naive Bayes algorithms

***Algorithms with Best Performance***

The k-Nearest Neighbor Classifier outperformed the Decision Tree Classifier with the best F1 scores and Accuracies on the testing sets:

    k-Nearest Neighbor Classifier
    -----------------------------
    * Accuracy: ~ 57%
    * F1 Score for 'Very Popular': 0.68
    * F1 Score for 'Moderately Popular': 0.54
    * F1 Score for 'UnPopular': 0.54
    
    Decision Tree Classifier
    ------------------------
    * Accuracy: ~ 57%
    * F1 Score for 'Very Popular': 0.68
    * F1 Score for 'Moderately Popular': 0.53
    * F1 Score for 'UnPopular': 0.52
    
***Evaluation after Optimization***

After tuning and optimization, we uncovered a huge improvement in the performace of the k-Nearest Neighbor Classifier. The accuracy of the k-Nearest Neighbor Algorithm increased by a nearly whopping 10%. On the other hand, the Dscision Tree Classifier had a mild improvements in performance. However, for each of the algorithms, the tuned model continued to massively overfitted the training dataset whilst improving the accuracy on the test set data.

***Algorithms for Use in Predictive Model***

We determined that the k-Nearest Neighbor Classifier and Decision Tree had the highest accuracy so we utilized them to perform our predictive model. Based on our results, we determined that the features in our dataset do not give us a complete understanding of the entire music industry as we would have liked it to. Given this reasoning, it seems fair to assume that either of these classifiers can be used to yield a high accuracy rate on the given dataset.
   

### Our Original Research Questions

***Are there specific features that directly correlate to song popularity?***

No, none of the feature variables had a strong positive or negative correlation to song popularity.

***Which feature variables are more indicative of projected song popularity?***

Using RFE, we conducted a feature selection and found that acousticness, danceability, energy, liveness, and loudness were the most indicative features of song popularity. 

***Will the accuracy of our predictions increase when we recursively eliminate features? Will certain algorithms perform better when there are more features in evaluation?***

We tried testing it on 4, 5, 6, and 7 features and found that 6 features resulted in algorithms that performed better. 

***A popularity of a song is dependent on a vast number of features that can be very difficult. Given the features that we do take into account, will knn work significantly better than a Decision Tree Classifier or a Support Vector Machine Classifier?***

After performing cross-validation, we found that the accuracy of k-Nearest Neighbors was slightly higher than the Decision Tree Classifier. 


***Reflection on our Findings***

Going back to our original project topic, we wanted to determine whether there were any relationships between our feature variables and song popularity. We were surprised to find that there were no strong correlations. 

However, when conducting our model, we actually found that when using six feature variables, there was a fairly high accuracy of 60% using the k-Nearest Neighbor algorithm. 


### Conclusion

One way to have more accurate results would be to increase the number of feature variables. We only used feature variables that were given to us by Spotify, but if we added more variables such as social media response time to songs or number of tweets about a specific song, we may have found a higher accuracy for our predictive model.

In addition to this, when looking up past research, we found many predictability models that used regression instead of classification. We would be curious to see if we kept song_popularity as a continuous variable, instead of classifying it into 3 buckets, would accuracy be higher? This is something we’d love to conduct in the future.

#### References
* https://blog.salesflare.com/spotify-iconic-product
* https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md
* https://techxplore.com/news/2019-09-spotify-songs.html
* https://www.digitaltrends.com/music/apple-music-vs-spotify/
* https://medium.com/@RyanJosephHill/why-spotify-is-one-of-my-favorite-products-93fa4dff850a
