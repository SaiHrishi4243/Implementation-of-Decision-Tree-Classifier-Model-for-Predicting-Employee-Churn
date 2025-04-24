# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sai Hrishi M
RegisterNumber:  212224240140


import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### data.head()

![image](https://github.com/user-attachments/assets/98eef73c-2479-44cd-97f1-3ac46bb7d1a1)

### data.info()

![image](https://github.com/user-attachments/assets/2e8d346f-4516-4b92-9980-50dbff896d94)

### data.isnull().sum()

![image](https://github.com/user-attachments/assets/564a2a51-bd21-47bf-839e-11e6c7aea295)

### data.value.count()

![image](https://github.com/user-attachments/assets/2003a4f8-7cb3-4a25-9f6e-1b976ceae554)

### x.head()

![image](https://github.com/user-attachments/assets/d113439b-4eea-4ee8-9eba-97399f9170e8)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
