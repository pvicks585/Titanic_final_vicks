import numpy as np 
import pandas as pd 

# Algorithms

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import re


train_df = pd.read_csv('/Users/paulvicks/Documents/train.csv')  #Create a variable for the training data
test_df = pd.read_csv('/Users/paulvicks/Documents/test.csv')    #Create a variable for the test data


total = train_df.isnull().sum()                                                     #Assign total to the number of null values per variable
missing_percent1 = total / train_df.isnull().count() * 100                          #Divide the total amount of null values by the number of rows for that variable
missing_percent2 = (round(missing_percent1, 2))                                       #Round off the percent above to two decimal places
missing_data = pd.concat([total, missing_percent2], axis=1, keys=['Total', '%'])    #Create a new dataframe that has a column for total and missing percentage
missing_data = missing_data.sort_values(by=['Total', '%'], ascending=False)         #Sort the missing_data dataframe in ascending order

train_df = train_df.drop(['PassengerId'], axis=1)  #Drop passenger ID as it does not effect probability of survival

#HANDLING NULL VALUES

deck = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 8}  #Create a dict for cabins called deck to map numerical values to
data = [train_df, test_df]   #Create a variable called data that houses both the training, testing set so that you can iterate and replace values easily

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")    #Have null values equal to "U0" so we can compile with regular expressions
    dataset['Deck'] = dataset['Cabin'].map(lambda x:    #Have the new variable column deck be equal to the capital letter at the beggining of cabin number
        re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)         #Map the dict deck from above to the new variable column 'Deck' in both datasets
    dataset['Deck'] = dataset['Deck'].fillna(0)         #Fill NA values with zero in Deck
    dataset['Deck'] = dataset['Deck'].astype(int)       #Have the values of 'Deck' be integers
    
train_df = train_df.drop(['Cabin'], axis=1)  #Now that the variable Deck is created, we can drop Cabin
test_df = test_df.drop(['Cabin'], axis=1)    #Drop the variable in both datasets

data = [train_df, test_df]

for dataset in data:
    mean_age = train_df["Age"].mean()    #Create a variable that is the mean of age
    std_age = test_df["Age"].std()       #Create a variable that is the standard dev of age
    is_null = dataset["Age"].isnull().sum()  #Find the amount of null variables in the variable Age
    random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = is_null)  #Create a variable that produces a random number between mean - std and mean + std for the size of missing values in Age
    age_copy = dataset["Age"].copy()   #Create a age copy variable to manipulate later
    age_copy[np.isnan(age_copy)] = random_age  #Have the null values of age in age_copy be equal to the values of the random age variable above
    dataset["Age"] = age_copy.astype(int)  #Have the variable age be equal to age_copy that we created above


train_df['Embarked'].describe() #SHows that S is the most common port of embarkment
default_embarked = 'S'  #Set default embarked to S
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(default_embarked)  #Have the null values of Embarked be equal to Port S


#TRANSFORMING CATEGORICAL VARIABLE TO NUMERICAL 

    
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)  #Fill in null values of fare (there was one) with zero and turn each value to an integer
 
titles = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}  #Create a dict for titles to map later
data = [train_df, test_df]

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  #Get the titles out of everyone's name
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')  #Have miscallaneous variables be labeled as rare
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')  #Label Mlle as Miss
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')    #Label Ms as Miss
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')    #Label Mme as Mrs
    dataset['Title'] = dataset['Title'].map(titles)   #Map the dict created above to the titles so that each title is assigned a number 1-5
    dataset['Title'] = dataset['Title'].fillna(0)  #Fill any NA's as 0
    
train_df = train_df.drop(['Name'], axis=1)  #You can now drop the variable column Name from the dataset
test_df = test_df.drop(['Name'], axis=1)


gender = {'male': 0, 'female': 1}    #Have males equal to zero, femals equal to 1
data = [train_df, test_df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(gender)  #for both datasets apply the gender dict above so each male is assinged 0 and female 1
    

train_df = train_df.drop(['Ticket'], axis=1)  #There are too many unique tickets (over 600), therefore we will drop it as a column
test_df = test_df.drop(['Ticket'], axis=1)


data = [train_df, test_df]
ports = {'S': 0, 'C': 1, 'Q': 2}  #Assign a number for each port

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)  #Use the dict above to map each number for each port to both datasets
    
data = [train_df, test_df]
for dataset in data:    #Assign a number to each age group, making sure a age bin doesn't contain a majority of the data
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 17), 'Age'] = 1  
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 23), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 23) & (dataset['Age'] <= 30), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 45), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 60), 'Age'] = 5
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 6
    

data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']  #Add together siblings per spouse and parent per children to get number of relatives
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0  #Assign 0 to those not alone
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1  #Assign 1 to those alone
    dataset['not_alone'] = dataset['not_alone'].astype(int)   


data = [train_df, test_df]
cut = pd.qcut(train_df['Fare'], 4)  #Cut the fares into 4 equal intervals of quantiles (in terms of each interval holds equal amount of data)
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.0, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.0) & (dataset['Fare'] <= 14.0), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.0) & (dataset['Fare'] <= 31.0), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)  #convert to integer
    



data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']   #Create age per class, as the older you are the better class you may buy 
    

data = [train_df, test_df]
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1) #Create fare per person for better accuracy of true fare
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    

#MACHINE LEARNING MODELING
    
X_train = train_df.drop("Survived", axis=1)  #Drop survived from x training set
Y_train = train_df["Survived"]  #Have y training set just be survived
test_pass_id = test_df['PassengerId']  #Assign to a new variable the test datasets passenger ID
X_test  = test_df.drop("PassengerId", axis=1).copy()  #Drop Passenger ID as it was dropped in the training set
    
    
#RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)  #Fit data to random forest

Y_prediction = random_forest.predict(X_test)  #Have random forest predict for the Y dataset

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2) #Calculate random forests accuracy
print(round(acc_random_forest,2,), "%") 


#K NEAREST NEIGHBOR
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)  #Fit training set to KNN with three centroids

Y_pred = knn.predict(X_test)  #Use KNN to predict survival

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)  #Calculate accuracy
print(round(acc_knn,2,), "%")

#CROSS VALIDATION
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=5, scoring = "accuracy")
    
print("Scores:", scores)
print("Mean:", scores.mean())

variable_strength = pd.DataFrame({'variable': X_train.columns, 'strength':random_forest.feature_importances_})
variable_strength = variable_strength.set_index('variable').sort_values('strength', ascending=False) #To see what variables hold weight
#not_alone holds no value
#We will drop that variable

train_df  = train_df.drop("not_alone", axis=1) #Drop not_alone from both datasets
test_df  = test_df.drop("not_alone", axis=1)


#RE RUN RANDOM FOREST WITHOUT NOT_ALONE
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")  #Print accuracy score



#For kaggle submission
submission = pd.DataFrame({
    'PassengerId':test_pass_id,
    'Survived':Y_prediction })

submission.to_csv('csv_to_submit.csv', index = False)
