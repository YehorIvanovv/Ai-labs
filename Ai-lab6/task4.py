import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

df = df.dropna(subset=['price', 'train_class', 'train_type', 'origin', 'destination', 'fare'])

le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_type = LabelEncoder()
le_fare = LabelEncoder()

df['origin_enc'] = le_origin.fit_transform(df['origin'])
df['dest_enc'] = le_dest.fit_transform(df['destination'])
df['type_enc'] = le_type.fit_transform(df['train_type'])
df['fare_enc'] = le_fare.fit_transform(df['fare'])

X = df[['price', 'origin_enc', 'dest_enc', 'type_enc', 'fare_enc']]
y = df['train_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Gaussian Naive Bayes Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
