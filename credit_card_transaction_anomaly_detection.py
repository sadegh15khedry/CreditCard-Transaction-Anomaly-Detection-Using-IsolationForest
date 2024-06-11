import pandas as pd
import matplotlib as mp
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import plot, show

#Loading the dataset
df = pd.read_csv("drive/MyDrive/creditcard_2023.csv")

#Finding and Handling missing values
null_values = df.isnull()
rows_with_null_values = df[null_values.any(axis=1)]
print(rows_with_null_values)

#normalizing the data using The z-score method
for column in df.columns:
    df[column] = (df[column] -
                           df[column].mean()) / df[column].std()







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

