import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

#Features inputs
X = df[["area","bedrooms"]]

#Label outputs
y = df["price"]

#train the model
model = LinearRegression()
model.fit(X,y)

new_data = pd.DataFrame({
    "area": [2200],
    "bedrooms": [3]
})

#make predictions
predictions = model.predict(new_data)
print('Predicted price', predictions)

