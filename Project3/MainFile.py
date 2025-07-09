import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#starting with task1:
#loading the downloaded dataset from kaggle and then loading to the project:
df = pd.read_csv("Titanic-Dataset.csv")

#task 2: handling the missing data
#we notice cabin has lot of missing values, so we can drop it using .drop function
df = df.drop(columns=["Cabin"])

#filling missing age values with median to avoid outlier issues
df["Age"] = df["Age"].fillna(df["Age"].median())

#filling embarked column values by mode as it's categorical
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#task 3 - normalisation
#as explained in videos, we normalise data between 0 and 1
df["Age_Norm"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
df["Fare_Norm"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())

#task 4 - encoding categorical data
#encoding 'Sex' as 0 and 1, and 'Embarked' as 0,1,2
df['Sex_encode'] = df['Sex'].map({'male': 0, 'female': 1})
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked_encode'] = df['Embarked'].map(embarked_mapping)
df.drop(['Sex', 'Embarked'], axis=1, inplace=True)  # dropping original columns

#starting with project 4: logistic regression model building
#selecting relevant features for model
features = ['Pclass', 'Age_Norm', 'Fare_Norm', 'Sex_encode', 'Embarked_encode']
X = df[features]    # input features
y = df['Survived']  # target variable

#splitting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#predicting and evaluating the model
y_pred = model.predict(X_test)

#printing accuracy and classification metrics
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#plotting confusion matrix using seaborn heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

'''Using logistic regression and scikit-learn, we were able to build a model that predicts Titanic passenger
survival with a reasonable accuracy (~80%). The analysis confirmed that survival was influenced most by
gender, passenger class, and fare. The model performed better at predicting non-survivors than survivors,
but overall showed clear learning from the data and confirmed known historical patterns'''

# Confusion Matrix Explanation:
# This matrix tells us how well our model is predicting survival.
# It shows 4 values:
# - True Positives (TP): People who actually survived and the model correctly predicted survival.
# - True Negatives (TN): People who did not survive and the model correctly predicted they wouldn’t.
# - False Positives (FP): People who did not survive but the model wrongly predicted they did.
# - False Negatives (FN): People who actually survived but the model wrongly predicted they didn’t.

# So, our model is good at predicting non-survivors,
# but it misses some actual survivors.
# Overall, it shows a decent performance, especially with around 80% accuracy.
