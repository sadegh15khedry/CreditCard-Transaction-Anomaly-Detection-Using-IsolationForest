import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, show
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
plt.style.use('ggplot')

#Loading the dataset
df = pd.read_csv("drive/MyDrive/CreditCardTransaction.csv") #Binary label indicating whether the transaction is fraudulent (1) or not (0)

#selecting proper column
df = df[["Year", "Month", "Department", "Division", "Merchant", "TrnxAmount"]]

#Exploratory data analysis
print(df.shape)
print(df.head())
print(df.describe())
print(df.info())
print(df.dtypes)
print(df.columns)
print("null values")
print(df.isnull().sum())
print("duplicated")
print(df.loc[df.duplicated()])

print(df["TrnxAmount"].value_counts())
print(df["Division"].value_counts())

print(df["Merchant"].value_counts())

ax = df["TrnxAmount"].plot(kind='hist')
ax.set_xlabel("TrnxAmount")
ax.set_ylabel("Count")
ax.ArtistList
plt.show()



df["Department"].value_counts().plot(kind='bar')

df["Division"].value_counts().plot(kind='bar')


ax = df.plot(kind='scatter', x='Merchant', y='TrnxAmount', colormap='viridis', alpha=0.3)
ax.set_xlabel("Merchant")
ax.set_ylabel("TrnxAmount")
ax.set_title("Merchant vs TrnxAmount")
plt.show()

#feature Relationship
sns.scatterplot(data=df, x="Merchant", y="TrnxAmount")
plt.show()

sns.pairplot(data=df, vars=["Year", "Month", "Department", "Division", "Merchant","TrnxAmount"])
plt.show()

print(df.corr())

sns.heatmap(df.corr(), annot=True)
plt.show()

#Finding and Handling missing values

df = df.dropna()



#Categorical Encoding 
label_encoder = LabelEncoder()
for column in ['Department', 'Division', 'Merchant']:
    df[column] = label_encoder.fit_transform(df[column])


#normalizing the data using The z-score method
for column in df.loc[:, df.columns[:-1]]:
    df[column] = (df[column] -
                           df[column].mean()) / df[column].std()



#Shuffling the dataset
df = df.sample(frac=1)


#Isolation tree
model_IF = IsolationForest(contamination='auto', random_state=33)
model_IF.fit(df)
label = model_IF.predict(df) #Returns -1 for outliers and 1 for inliers.
score = model_IF.decision_function(df)

df['Label'] = label
df['Score'] = score

print(df)


#Visualizing 
df = df.sample(frac=.001)

df['Label'].plot(kind='hist')
plot(df['Department'], df['Division'], df['Label'], 'go')
for label in df['Label']:
  subset = df[df['Label'] == label]
  if (label == -1):
      plt.scatter(subset['Department'], subset['Division'], s=100, c="red")
  elif label == 1:
      plt.scatter(subset['Department'], subset['Division'], s=100, c="blue")


plt.title('Anomaly')
plt.xlabel('Department')
plt.ylabel('Division')
plt.legend()
plt.show()
