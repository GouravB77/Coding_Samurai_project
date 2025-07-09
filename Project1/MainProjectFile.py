import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#I have divided the project into different stages for better understanding
#Stage 1: Loading and getting a brief summary of the dataset
#loading dataset:
df=pd.read_csv('sales_data_sample.csv',encoding='latin1')

#displaying rows,columns and other attributes:
print("First 5 rows are:")
print(df.head(),"\n")

print("The shape of the dataset is:", df.shape,"\n")

print("The columns present in the dataset are:")
print(df.columns.tolist(),"\n")

print("Info about the loaded dataset:")
print(df.info(),"\n")

print("Important stats about the dataset are as follows:")
print(df.describe(),"\n")



#Step2: Data cleaning
#Getting number of missing values first to analyse
print("Missing values for every column:")
print(df.isnull().sum())

#removing any duplicate rows
print("Number of duplicate rows:",df.duplicated().sum())
df=df.drop_duplicates()

#As seen from the columns the dataset has no date-time columns, so need for conversion there
#We observe that the dataset is comparatively clean and requires no adjustment
#We proceed to fill the missing values in code:
df['ADDRESSLINE2'] = df['ADDRESSLINE2'].fillna('Unknown')
df['STATE'] = df['STATE'].fillna('Unknown')
df['POSTALCODE'] = df['POSTALCODE'].fillna(df['POSTALCODE'].mode()[0])
df['TERRITORY'] = df['TERRITORY'].fillna(df['TERRITORY'].mode()[0])

print(df.isnull().sum())
#we have filled all the missing values now


#Visualizing and analyzing:
plt.figure(figsize=(12,5))
sns.histplot(df['SALES'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales-Amount')
plt.ylabel('Frequency')
plt.show()
'''By this histplot, we see it is right skewed and hence there are 
lesser high value orders and more low value orders'''

#Analysing the top products:
top = df.groupby('PRODUCTCODE')['QUANTITYORDERED'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top.values, y=top.index)
plt.title('Top 10 Products based on Quantity')
plt.xlabel('Quantity Ordered')
plt.ylabel('Product Code')
plt.show()
#By this barplot, we observed which product is the most bought



territory_sales = df.groupby('TERRITORY')['SALES'].sum()
plt.figure(figsize=(10,10))
plt.pie(territory_sales, labels=territory_sales.index)
plt.title('Sales Distribution by Territory')
plt.show()
#by this pie chart, we observe in which territory the max products are sold
#project 1 complete







