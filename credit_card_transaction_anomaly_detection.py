import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, show
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import seaborn as sns
import numpy as np
plt.style.use('ggplot')

#Loading the dataset
df = pd.read_csv("drive/MyDrive/creditcard_2023.csv")



#Exploratory data analysis
# print(df.shape)
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.dtypes)
# print(df.columns)
# print("null values")
# print(df.isnull().sum())
# print("duplicated")
# print(df.loc[df.duplicated()])

# print(df["Amount"].value_counts())
# print(df["Class"].value_counts())
# print(df["V1"].value_counts())

# ax = df["Amount"].plot(kind='hist')
# ax.set_xlabel("Amount")
# ax.set_ylabel("Count")
# ax.ArtistList
# plt.show()

# ax = df["V2"].plot(kind='hist')
# ax.set_xlabel("V2")
# ax.set_ylabel("Count")
# ax.set_title("V2")
# plt.show()


# df["Class"].value_counts().plot(kind='bar')
# df["V1"].value_counts().plot(kind='bar')


# df["Amount"].plot(kind='hist')
# df["Class"].plot(kind='hist')
# df["V1"].plot(kind='hist')

ax = df.plot(kind='scatter', x='V1', y='Amount', c='Class', colormap='viridis', alpha=0.3)
# ax.set_xlabel("V1")
# ax.set_ylabel("Amount")
# ax.set_title("V1 vs Amount")
# plt.show()

# #feature Relationship
# sns.scatterplot(data=df, x="V1", y="Amount", hue="Class")
# plt.show()

# sns.pairplot(data=df, vars=["V1", "V2", "V3", "V4", "Amount"], hue="Class")
# plt.show()

# print(df.corr())

# sns.heatmap(df.corr(), annot=True)
# plt.show()

#Finding and Handling missing values
# null_values = df.isnull()
# rows_with_null_values = df[null_values.any(axis=1)]
# print(rows_with_null_values)



#normalizing the data using The z-score method
for column in df.columns:
    df[column] = (df[column] -
                           df[column].mean()) / df[column].std()



#Shuffling the dataset
df = df.sample(frac=1)

#Spliting the dataset into test, train, eval
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns[:-1]], df["Class"], test_size=0.2, train_size=0.8)


#Isolation tree
model_IF = IsolationForest(contamination=0.5, random_state=33)
model_IF.fit(x_train)
train_pred = model_IF.predict(x_train)
test_pred = model_IF.predict(x_test)


print(test_pred)

# print(x_train.shape)
# print(x_cv.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_cv.shape)
# print(y_test.shape)



#Visualizing the anomalies
# df['label'].plot(kind='hist')
# plot(df['Age'], df['Annual Income (k$)'], df['label'], 'go')
# for label in df['label']:
#   subset = df[df['label'] == label]
#   if (label == 0):
#       plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="red")
#   elif label == 1:
#       plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="blue")
#   elif label == 2:
#       plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="green")



# plt.title('Customer Segments')
# plt.xlabel('Age')
# plt.ylabel('Annual Income (k$)')
# plt.legend()
# plt.show()

