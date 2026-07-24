#libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier #for KNN MODEL
from sklearn.metrics import confusion_matrix,f1_score, classification_report

#load datasets
iris = load_iris()

#feature and target 
X = iris.data          #sepal and petal length + width 
Y = iris.target 
print("Total Samples:" ,X.shape[0])
print("Total features:", X.shape[1])
print("classes:",iris.target_names)

# feature scaling
scalar  = StandardScaler()
X_scaled =scalar.fit_transform(X)

#train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_scaled,Y,test_size =0.2,random_state=42,shuffle=True)

model = KNeighborsClassifier(n_neighbors=5) #KNN model with 5 neighbors

model.fit(X_TRAIN,Y_TRAIN) #train the model

Y_PRED = model.predict(X_TEST) #predict the test data

cm=confusion_matrix(Y_TEST,Y_PRED) #confusion matrix
f1=f1_score(Y_TEST,Y_PRED,average='macro') #f1 score

print("\nConfusion Matrix:\n")
print(cm)
print("\nF1 Score:",round(f1,3))  
print("\nFULL REPORT:\n",classification_report(Y_TEST,Y_PRED,target_names=iris.target_names)) #full report