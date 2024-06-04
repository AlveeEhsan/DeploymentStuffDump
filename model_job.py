import joblib
print("jrllo")
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# Load the iris dataset
iris=load_iris()

x= iris.data
y=iris.target

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(x,y)

# Save the trained model
joblib.dump(model,'model.joblib' )