import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#starting with task1:
#loading the downloaded dataset from kaggle and then loading to the project:
df=pd.read_csv("Titanic-Dataset.csv")

#displaying the first 5 rows using head function:
print(df.head())

print("")

#using info and describe functions:
#info tells 1.columns 2.non null value count  3.count(tells null values)  4.dtype-column ki data type
print(df.info)

print("")

#tells 1.count(rows in each column here)  2.max,min 3.percentiles 4. mean,standard dev
print(df.describe())


#identifying columns with missing values in each column:
print(df.isnull().sum())

#task 1 completed



#starting with task 2: handling the missing data
#if lot of data, better to remove few missing values or drop the column as it does not contribute to modelling that much
#if lot of missing values then we try to fill the values (eg-replace by mean, median,mode etc.)

#we notice cabin has lot of missing values, so we can drop it using .drop function
df=df.drop(columns=["Cabin"])

#we notice age also has many missing values(not that much as cabin to be dropped off)
#as stated in the task, we can replace age column missing values by mean or median, here I am replacing by the median
# as explained in the video by campusX, median is always better than mean due to outlier problem

df["Age"]=df["Age"].fillna(df["Age"].median())

#filling embarked column values by mode:
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
#.mode returns a series as there can be more than one mode and we chose the first mode

#to check for missing values apply the isnull func:
#print(df.isnull().sum())


#task 2 finished




#starting task3- normalisation and stand.
#as in the videos, normalise data between 0 and 1 or standardise(mean is 0 and sd=1)
#here we have taken the normalised approach
df["Age_Norm"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Fare_Norm"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())

#checking the values using head:
print(df[["Age", "Age_Norm"]].head())
print(df[["Fare", "Fare_Norm"]].head())

#checking by bar chats:
plt.figure(figsize=(12, 4))


#plotting the original values:
plt.subplot(1, 2, 1)
plt.bar(df.index[:10], df["Age"][:10])
plt.title("Original Age")
plt.xlabel("Index")
plt.ylabel("Age")

#plotting the normalised ages
plt.subplot(1, 2, 2)
plt.bar(df.index[:10], df["Age_Norm"][:10])
plt.title("Normalised Age")
plt.xlabel("Index")
plt.ylabel("Changed Age")

plt.show()

#task 3 complete


#starting task 4-encoding categorical data
# as seen in the CoderArts video, encoding is required so that the Ml model can efficiently understand the values as numerical values are always better to compare
#for sex, assigning 0 and 1 and for Embarked using binary can lead to some undefined states, using lettered one hot encoding for it

df['Sex_encode'] = df['Sex'].map({'male': 0, 'female': 1})

#endoding 0,1,2 for C,Q,S:
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked_encode'] = df['Embarked'].map(embarked_mapping)

df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
#axis=1 is used to drop column, axis=0 for row

print(df.head())
#task 4 completed



#starting with task 5-visualization portion
#countplots tells frequency of each unique category in a particular column and for this we use seaborn library
sns.countplot(data=df,x='Pclass')
plt.title('Number of Passengers by Passenger Class')
plt.xlabel('P Class')
plt.ylabel('Count')
plt.show()
'''we can infer from here that majority of passengers were in  3rd class'''
sns.countplot(x='Pclass',hue='Survived',data=df)
plt.title('Survival Count on basis of Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No','Yes'])
plt.show()
'''This confirms that passengers in 1st class had higher survival rates owing to class distinction 
and discrimination issues'''



sns.countplot(data=df,x='Sex_encode')
plt.title('Number of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
'''This tells us that there were greater male passengers(1) than female passengers'''
sns.countplot(x='Sex_encode',hue='Survived',data=df)
plt.title('Survival Count on basis of Gender')
plt.xlabel('Gender 0=Male 1=Female')
plt.ylabel('Count')
plt.legend(title='Survived',labels=['No','Yes'])
plt.show()
'''This tells that females had higher survival rate and confirms that females were rescued first'''


sns.countplot(data=df,x='Survived')
plt.title('Number of Passengers by Survival')
plt.xlabel('Survived 0=No,1=Yes')
plt.ylabel('Count')
plt.show()
'''This tells us that there were lesser number of survivors out of the total population
Also, the survival rate is less than 50%'''


#starting off with the pie charts:
df['Embarked_encode'].value_counts().plot.pie()
plt.title('Port of Embarkation Distribution(0,1,2 for C,Q,S Respectively)')
plt.show()
'''This tells us most passengers embarked from S port'''


#plotting heatmap for showing correlation between fare and survival
num_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(num_df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()
'''This tells us that more you paid, better was your chance of survival'''

#Plotting histplot to show age bracket division:
sns.histplot(df["Age"],kde=True,bins=30)
plt.title('Age Bracket Division of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
'''We can infer from here that max passengers from age 20-40'''
'''Project 2 (task3 in sheet) Completed'''















