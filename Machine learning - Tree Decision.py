from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#dataset de carros
dataset = pd.read_csv("Carro_dataset_novo_csv.csv", header=None)
#label do dataset
labels = pd.read_csv("Carro_dataset_labels.csv", header = None)
#Print do formato dos dados
print(dataset.shape)
print(labels.shape)
#"Average_Price" , "Maintain" , "Doors" , "Capacity" , "Luggage" , "Safety" , "Result" ]


X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size= 0.1)
#Classificador tree decision
my_classifier = tree.DecisionTreeClassifier()
#Testando
my_classifier.fit(X_train, y_train)
#Prevendo
predictions = my_classifier.predict(X_test)
#Precisão
print(accuracy_score(y_test, predictions))


