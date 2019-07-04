from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

features = ["Average_Price" , "Maintain" , "Doors" , "Capacity" , "Luggage" , "Safety" , "Result" ]
label = ["Label"]

dataset_cars = pd.read_csv("Carro_dataset_novo_csv.csv", header = None)
labels = pd.read_csv("Carro_dataset_labels.csv", header=None)


X_train, X_test, y_train, y_test = train_test_split(dataset_cars, labels, test_size = 0.1)
#Classifidores K = 3
classifier_knn = KNeighborsClassifier(n_neighbors=3)
#treinando os dados
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_train)
#previsão
a = metrics.accuracy_score(y_train, y_pred)
#recisão
print(a)


