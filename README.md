# Tanzanian Water Pumps: How to find pumps in need of repair 

Water pumps are an essential tool in developing nations to deliver safe and reliable water to rural communities and have become a hallmark of the United Nations’ Sustainable Development Goals (SDGs) to drive down poverty globally. The typical water pump taps into below ground water sources, which tend to be less contaminated than surface water sources and allow water-scarce communities access to a vital resource. While water pumps have been a focal point for development programs, it is estimated that between //$1.2bn to //$1.5bn in investment were lost from 1989 to 2009 from unfunctional wells, according to non-profit [Lifewater.org] ((https://lifewater.org/blog/3-reasons-water-wells-fail-and-why-sustainable-development-is-possible/) . Determining which water pumps are in need of repair before they become nonfunctional can both help save development programs from costly replacement and, more importantly, ensure that vulnerable communities are not left without clean drinking water. 

In the following project, we examined data collected by [Taarifa]((http://taarifa.org/)an infrastructure open source platform, and the [Tanzanian Water Ministry](http://maji.go.tz/) and published by [Driven Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) on nearly 60,000 water pumps across Tanzania. The data encompassed 39 different characteristics of the pumps including location data, pump management information, and pump characteristics. We used the data in an iterative modeling process to create a final classification model that could be used by development organizations to predict whether or not a pump is in need of repair before the pump fails.

In this repository you can find the following: 
* Data files 
* Images
* Modeling Notebook
* Jupyter Notebook
* Presentation
* ReadMe 

# Data and Model Limitations 
While our model serves as a predictor for the status of water pumps across Tanzania, it is important to address its limitations up front. First, data quality proved to be difficult. Many of the indicators had missing values, nonsensical inputs, or suffered from alternate spellings/abbreviations. This made it difficult to discern the true value of many inputs. For example, nearly a third of the data claimed to serve a population of 0. Further, our dataset suffered from an imbalance issue of our variable of interest - status_group-as  pumps classified as “functional, needs repair” accounted for only 7.2% of the observations. Finally, while our classification model has the best precision — or  the percentage of predictions of pumps that need repair that actually need repair — and improves on the base model, it can be improved. As discussed below, we believe more robust or standardized data collection methods and improved balancing modeling could help improve predictive capabilities. 

# Data Overview
Our data looked at 59,400 observations and 39 characteristics (columns)  on water pumps in Tanzania. 

Our target variable, status_group, represents the status of the water pump categorized as either ‘functional’, ‘functional needs repair’, and ‘non functional’ — 54% of observations were functional pumps, 38% non functional, and 7% of pumps were deemed functional, needs repair. 

We dropped columns that offered information about data collection, included repetitive information from other columns, or were found to be of low feature importance in our modeling. We also used the data to create three new indicators: construction_yr_missing, water_per_population, age_at_inspection. 

Our final model included information from 23 columns 59,399 observations. 

# Data Process - An Iterative Approach
In order to construct the best predictive model of water pumps that are in need of repair, we implemented an iterative modeling on training data where we started with a baseline model where the model would always predict the majority class (functional),  and moved to more complex models through implementing decision trees, K-nearest neighbors, and random forests. To determine the best hyperparameters within each model, we also conducted GridSearchCV. In order to determine the best predictive model, we compared precision rates across models. We determined that this was the best metric to set as our determinant because when looking at water pumps we felt it is most important that we have a model that gives us the highest confidence that when we predict a pump that needs repair, the pump actually needs repair. This is due to the time and financial constraints it would cause a nonprofit like Water.Org to visit each pump. 

In general we followed the following process after creating our training data set:
 
* Transformed our categorical, numerical, and ordinal columns through ColumnTransformer and instantiated the model using Pipeline. 
* Fit our model on the training set. 
* Cross validated on the training set. 
* Predicted y values from the testing data based on the model
* Examined the classification report

In the case of K-nearest neighbors and random forest classifiers  we also ran feature importance to see if there were features we could drop in our next iteration to strengthen our model. 

# Dummy Model — Always Predict Majority Class
Our dummy classifier,  the classification model that the rest of our models would be compared, we set to always pick the majority class. In this case the model would also predict that a pump would be functional and have an accuracy rate of 54%. For our process, this means that the model would never predict that a pump is in need of repair and thus, is functionally useless.  

# Model 1: Basic Logistic Regression 
Running a basic logistic regression of all our variables of interest lead to an accuracy rate of 0.75, meaning that our model would predict the correct status group of water pumps only 75% of the time. Our precision rate for our group of interes  — pumps needing repair — was .51. Meaning that 51% of the time that our model predicts a functional pump needing repair, the pump was actually a functional pump needing repair. While an improvement on the dummy regressor, we believe that running other models would yield higher accuracy and precision rates.  

# Model 2: K-Nearest Neighbors – High Accuracy, Low Precision 
We then ran a series of grid searches using the k-nearest neighbors classifier. Our grid search involved a variety of hyperparameters including n_neighbors, weights, and metric. The grid search revealed that the best estimator for classification had the following hyperparameters: 

* Metric: Manhattan
* N_neighbors: 17
* Weights: distance 

While the model had the highest accuracy scores of the gridsearch (and all of our models) at 0.94, its precision for predicting pumps that are functional needs repair was worse than our logistic regression model at 0.48 - meaning that 48% of the time  our model predicts a pump needing repair, the pump actually needs repair. 

# Model 3: Best Model Random Forest Classification
In order to ensure that we are not overfitting to our training data, we employed the random forest classifier. We ran a basic random forest model with default random forest hyperparameters. This model, like the KNN model, had a high accuracy rate of 0.94 but low precision for our class of interest with 0.46 – the lowest so far of our models. 

Acknowledging the power of random forest classifier we ran a gridsearch adjusting the n_estimators, min_samples_leaf, and criterion. From this gridsearch we found the optimal hyperparameters of our search were the following:
* N_estimators : 100 (the default)
* Min_leaf_samples: 5
* Criterion: gini

The accuracy of this model was lower than the individual trees at 0.80, however, our precision was the highest of all of our models at .68 — meaning that 68% of the time our model predicts a water pump in need of repair, the water pump does need a repair. We took this model and ran feature_importances to find the best predictors of the class in this most precise model and found these three features had some of the highest predictors of classification within the model: 
* Whether or not the pump is in the Internal Drainage Basin
* Age of pump at inspection (a feature that we engineered using construction year and recorded date)
* Gps_height or the altitude of the pump. 
* Whether the quantity of the well was insufficient or not. 

We then removed the least important features (“lga”, “management”, and “region_code”). However this model had the same accuracy rate and precision, but a lower recall. So we determined that while unimportant these features were important in our classifier. 

## Model 3b: Random Forest with SMOTE 
Finally, we acknowledged that our data was very imbalanced in our class of interest and wanted to address this imbalance and see if it improved the performance of our best model classifier. When we ran a SMOTE where we over indexed the minority class, however, our accuracy and precision fell. This was a surprise and led us to believe that using a different SMOTE  strategy could have ultimately improved our classification and changed our best model.  

# Findings & Recommendations
Our model reveals a few things about pumps in need of repair in Tanzania:
* 1. Internal Water Basin. Whether or not a pump resides in the Internal Water Basin appeared to be a big indicator of its status in our final classification model. After looking further at the Internal Water Basin we find that it is the third largest basin in the country (out of 9). However, 80% of the population that resides in the area suffer from water scarcity. We recommend that if a nonprofit were looking to repair water pumps, that it start with pumps in this region. 
* 2. Altitude of the pump. The altitude of the pump also appeared to have high influence over the status of the pump in our final classification model. We found that pumps at higher altitude need to overcome lower pressure through different pump construction. When we looked at our data these pumps at higher altitudes were more likely to be functional. When repairing pipes, we recommend utilizing some of the techniques for higher altitude pumps. 
* 3. Age of pumps. Finally, our age of pumps at inspection also was an important feature in determining pump status in our final model. We find that the  average age of pumps that need repair is 17 years compared to 12 years for pumps not needing repair. This leads us to believe that pumps typically start to need repairs between 12  and 17 years and recommend checking more frequently on pumps in this age. 

On the data side we also recommend a few things based on our model: 
* 1. Standardization of data collection techniques. To improve our prediction tool we recommend that inspectors (whether local management or larger organizations) standardize data collection process and record keeping through a wide scale education program put on by the Tanzania Ministry of Water. As mentioned, the data suffered from missing, redundant, mispelled or nonsensical values (populations of 0). Improvement of data collection will allow more usable data — particularly in terms of pump age — and decrease the amount of imputation our model needs to undertake. 
* 2. Improve imbalance issues with a different SMOTE strategy. We know that the class of interest iss largely imbalanced in the data set. Our model both over-sampled the minority while keeping the majority classes the same and then oversampled the minority while undersampling the majority classes. However, both methods did not improve our classification. We recommend attempting other SMOTE strategies to address this imbalance.  
* 3. Pruning our random forest more to increase precision. Finally, while our final model had the best precision out of all of our models, it still leaves a lot of room for misclassification. We recommend tuning the model by testing different hyperparameters to see if precision improves. 

