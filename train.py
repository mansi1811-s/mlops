import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Sample dataset
data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 5],
    "price": [200000, 240000, 300000, 360000, 400000, 440000, 500000, 560000]
}

df = pd.DataFrame(data)

#Features inputs
X = df[["area","bedrooms"]]

#Label outputs
y = df["price"]

#split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

#train the model
model = LinearRegression()
model.fit(X_train,y_train)

#test model
score = model.score(X_test, y_test)
print("Model accuracy:", score)

new_data = pd.DataFrame({
    "area": [2200],
    "bedrooms": [3]
})

#make predictions
predictions = model.predict(new_data)
print('Predicted price', predictions)

