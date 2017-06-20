
# coding: utf-8

import matplotlib.pyplot as plt
import sklearn.ensemble as sk
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.svm import SVC


data = pd.read_csv('Data&Data Classification Challenge - Facebook - Training Set.csv', header=0, sep='\t')

data.head()

data.describe()

#percentage of non-null vlaues per column
data.count(0)/data.shape[0] * 100

data.found_keywords_occurrences.value_counts().head()

# Only 12% of the found_keywords column is filled -- We will not include it in our model

data.owner_type.value_counts()


##Check the categories' distribution in the dataset
byLabel = data.groupby('INDEX New').size()

my_plot = byLabel.plot(kind='bar')
plt.show()


# Analyze the posts by day/hour/month

#Convert the date field to date format
data['dateForm'] = pd.to_datetime(data.published_at)
#Create a month column
data['month'] = data['dateForm'].apply(lambda x: x.month)
#Create a day column
data['day'] = data['dateForm'].apply(lambda x: x.day)
#create an hour column
data['hour'] = data['dateForm'].apply(lambda x: x.hour)


# In[20]:

#Visualize distribution by month
bymonth = data.groupby(['month', 'INDEX New']).size()
bymonth.unstack(1).plot(kind='bar')


# Overall, the 3 categories have the same behavior when it comes to the month. There's an increase until month 5 and then a decrease till month 12.

# In[21]:

byday = data.groupby(['day', 'INDEX New']).size()
byday.unstack(1).plot(kind='bar')


# In[24]:

byhour = data.groupby(['hour', 'INDEX New']).size()
byhour.unstack(1).plot(kind='bar')


# In[22]:

byowner = data.groupby(['owner_type', 'INDEX New']).size()
byowner.unstack(1).plot(kind='bar')


# When it comes to differentiating between Reseller and Fake Seller, the owner type will be important.
# Pages are more likely to be resellers whereas users have higher records of being fake sellers

# ### Classification

# ##### Model 1: Naive Bayes

# We can start by trying with the Naive Bayes classifier. However, in order to use it, we need to first extract
# feature vectors for the text data in order to use it as input to the machine learning algorithm. We will
# start by choosing only the description column to see how accurate would the prediction be based
# on the post description.
# 
# To do this, we will start by extracting the feature vectors from the description value to
# be used in the machine learning algorithm.
# 

# Using Count vectorizer, we get the occurence count of words. However, the count does not account
# for word importance. Usually, this can be done using the tfidf algorithm, which will downscale the score
# for the words who appear often, and therefore will give more importance to the words that have significance
# but occur in small portions.
# 
# With the description column, we will be using the tfidf vectorizer

categories = data['INDEX New']
desc = data['description'].fillna('')

vectoriser = TfidfVectorizer() 
features = vectoriser.fit_transform(desc)

x, x_test, y, y_test = train_test_split(features,categories,test_size=0.2,train_size=0.8, random_state = 0)

clf = MultinomialNB().fit(x, y)
predicted = clf.predict(x_test)


def printreport(exp, pred):
    print(pd.crosstab(exp, pred, rownames=['Actual'], colnames=['Predicted']))

    print('\n \n')
    print(classification_report(exp, pred))


# In[33]:

printreport(y_test, predicted)


# With Naive Bayes, and using only the description column of the dataset we get around 0.79
# average accuracy and 0.78 on recall.

# Since we are dealing with an algorithm to detect fake product sales, it is
# desirable to identify fake or possibly fake products. We will focus on improving recall since
# it will measure the number of fake products that were detected.

# ##### Model 2: Random Forest 

# Convert owner to binary

data2 = data.copy()

owner = pd.get_dummies(data2['owner_type'])
data2 = pd.concat([data2, owner], axis=1)   

data2 = data2.fillna('')

mapper = DataFrameMapper([
     ('description', TfidfVectorizer()),
     ('nb_like', None),
     ('picture_labels', TfidfVectorizer()),
     ('nb_share', None),
     ('user', None),
     ('month', None),
     ('day', None),
     ('hour', None),
 ])

features = mapper.fit_transform(data2)
categories = data2['INDEX New']

# Split the data between train and test
x, x_test, y, y_test = train_test_split(features,categories,test_size=0.2,train_size=0.8, random_state = 0)

clf = sk.RandomForestClassifier(random_state=0)
clf.fit(x, y)

predicted = clf.predict(x_test)

printreport(y_test, predicted)

#######
# Clean Dataset
#######

# Now, we will try to improve the model by cleaning the dataset, removing and adding some features
# We will add a column to identify the userID based on the profile picture url. This information should be
# helpful especially when dealing with recurrent users who are known to be Resellers or fake sellers. We will
# also add a new column based on the picture url to identify the same picture as an item

# add a user id based on the profile picture
data2['userID'] = pd.factorize(data2.profile_picture)[0]

# Add item id based on the picture url
data2['itemID'] = pd.factorize(data2.pictures_url)[0]

# Only 5579 values for nb_share are != 0. We will remove it from the model and try again
len(data2[data2.nb_share != 0])

mapper = DataFrameMapper([
     ('description', TfidfVectorizer()),
     ('nb_like', None),
     ('picture_labels', TfidfVectorizer()),
     ('user', None),
     ('month', None),
     ('day', None),
     ('hour', None),
     ('userID', None),
     ('itemID', None),
 ])


features = mapper.fit_transform(data2)
categories = data2['INDEX New']

x, x_test, y, y_test = train_test_split(features,categories,test_size=0.2,train_size=0.8, random_state = 0)

clf = sk.RandomForestClassifier(random_state=0)
clf.fit(x, y)

predicted = clf.predict(x_test)

printreport(y_test, predicted)

# By adding the user ID and Item ID, we were able to increase the recall to 85% and accuracy to 86%
# The reseller category is still not properly identified by the model

clf = sk.RandomForestClassifier(random_state=0, n_estimators=70)
clf.fit(x, y)

predicted = clf.predict(x_test)

printreport(y_test, predicted)

# By increasing the number of estimators, we now have a recall of 88%, and the recall for Reseller has also increased
# from 76% to 81%

# If we take a look a look again at the graphs that we have at the beginning as part of the data understanding,
# we can see that almost date does not play a major role in differentiating between the categories.
# Therefore, we will try to remove the date data (month, hour ..) and run the model again

mapper = DataFrameMapper([
     ('description', TfidfVectorizer()),
     ('nb_like', None),
     ('picture_labels', TfidfVectorizer()),
     ('user', None),
     ('userID', None),
     ('itemID', None),
 ])

features = mapper.fit_transform(data2)
categories = data2['INDEX New']

x, x_test, y, y_test = train_test_split(features,categories,test_size=0.2,train_size=0.8, random_state = 0)


clf = sk.RandomForestClassifier(random_state=0, n_estimators=70)
clf.fit(x, y)

predicted = clf.predict(x_test)

printreport(y_test, predicted)


# Clean the picture label column: Remove duplicated labels, empty labels
data2.groupby('picture_labels').count()['description'].head(50)

data2['newLab'] = data2['picture_labels'].apply(lambda x: x.split(','))
data2['newLab2'] = data2['newLab'].apply(lambda x: list(set([y.strip() for y in x if y != ' ' and y != ''])))
data2['newLab3'] = data2['newLab2'].apply(lambda x: ','.join(x))
data2.groupby('newLab3').count()['description'].sort_values(ascending = False).head(20)

mapper = DataFrameMapper([
     ('description', TfidfVectorizer()),
     ('nb_like', None),
     ('newLab3', TfidfVectorizer()),
     ('user', None),
     ('userID', None),
     ('itemID', None),
 ])



features = mapper.fit_transform(data2)
categories = data2['INDEX New']

x, x_test, y, y_test = train_test_split(features,categories,test_size=0.2,train_size=0.8, random_state = 0)

clf = sk.RandomForestClassifier(random_state=0, n_estimators= 100)
clf.fit(x, y)

predicted = clf.predict(x_test)

printreport(y_test, predicted)

# Adding the cleaned label did not improve the accuracy or recall

# Final Model: For further improvements, we could look at feature importance for Random Forest and pick the
# top features and try again with the classification

