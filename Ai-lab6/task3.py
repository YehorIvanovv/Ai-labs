import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import warnings

warnings.filterwarnings("ignore")

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le_outlook = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook_enc'] = le_outlook.fit_transform(df['Outlook'])
df['Humidity_enc'] = le_humidity.fit_transform(df['Humidity'])
df['Wind_enc'] = le_wind.fit_transform(df['Wind'])
df['Play_enc'] = le_play.fit_transform(df['Play'])

X = df[['Outlook_enc', 'Humidity_enc', 'Wind_enc']]
y = df['Play_enc']

model = CategoricalNB()
model.fit(X, y)

variant_outlook = le_outlook.transform(['Sunny'])[0]
variant_humidity = le_humidity.transform(['High'])[0]
variant_wind = le_wind.transform(['Weak'])[0]

X_test = [[variant_outlook, variant_humidity, variant_wind]]
prediction_enc = model.predict(X_test)
prediction = le_play.inverse_transform(prediction_enc)
probabilities = model.predict_proba(X_test)[0]

print("Варіант 3: Outlook = Sunny, Humidity = High, Wind = Weak")
print(f"Чи відбудеться матч (Play)? -> {prediction[0]}")
print(f"Ймовірність 'No': {probabilities[0]:.2f}")
print(f"Ймовірність 'Yes': {probabilities[1]:.2f}")
