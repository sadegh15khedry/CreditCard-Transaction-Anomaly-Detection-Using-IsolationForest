# Anomaly Detection in Credit Card Transactions

This project implements anomaly detection in credit card transactions using the Isolation Forest algorithm. The dataset used includes transaction details such as year, month, department, division, merchant, and transaction amount.

## Getting Started

To run the code, follow the steps below:

### Prerequisites

- Python 3.x
- Required Python packages: pandas, matplotlib, seaborn, scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd https://github.com/sadegh15khedry/CreditCard-Transaction-Anomaly-Detection-Using-IsolationForest
   ```

2. Install dependencies:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

### Usage

1. Load the dataset:
   
   Get the dataset from kaggle using https://www.kaggle.com/datasets/ybifoundation/credit-card-transaction
   Ensure the dataset `CreditCardTransaction.csv` is placed in the appropriate directory or update the file path in the code (`pd.read_csv("drive/MyDrive/CreditCardTransaction.csv")`).

3. Run the code:

   ```bash
   python anomaly_detection_credit_card.py
   ```

4. Explore the results:
   
   - The code performs exploratory data analysis, preprocessing, applies Isolation Forest for anomaly detection, and visualizes anomalies detected.

## Code Structure

- `anomaly_detection_credit_card.py`: Main script that loads data, preprocesses it, applies Isolation Forest, and visualizes anomalies.
  
## Features

- **Data Loading**: Loads credit card transaction data from a CSV file.
  
- **Exploratory Data Analysis**: Analyzes data distribution, checks for missing values, and explores relationships between variables.
  
- **Preprocessing**: Handles missing values, encodes categorical variables using LabelEncoder, and normalizes numerical features using z-score method.
  
- **Isolation Forest**: Utilizes Isolation Forest algorithm for unsupervised anomaly detection.
  
- **Visualization**: Uses matplotlib and seaborn for visualizing data distributions, scatter plots, and anomalies detected.

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## Acknowledgments

- Inspired by tutorials and examples on anomaly detection and Isolation Forest.
