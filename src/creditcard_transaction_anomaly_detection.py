import matplotlib.pylab as plt
from matplotlib.pyplot import plot
from sklearn.ensemble import IsolationForest


def visualize_results(df):  
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

    
def detect_anomaly(model_IF, df):    
    
    label = model_IF.predict(df) #Returns -1 for outliers and 1 for inliers.
    score = model_IF.decision_function(df)

    df['Label'] = label
    df['Score'] = score

    print(df)
    return df


def get_Isolation_forest_model():
    model = IsolationForest(contamination='auto', random_state=33)
    return model
