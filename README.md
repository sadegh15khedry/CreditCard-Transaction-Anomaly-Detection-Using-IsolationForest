# Anomaly Detection in Credit Card Transactions
The Isolation Forest is a powerful tool for detecting fraudulent credit card transactions. Its unsupervised nature allows it to detect anomalies in datasets without requiring labeled data, making it particularly useful in situations where fraudulent transactions are rare and unpredictable. 

## Table of Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Results](#Results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- 

## Overview
This project implements anomaly detection in credit card transactions using the Isolation Forest algorithm. The dataset used includes transaction details such as year, month, department, division, merchant, and transaction amount.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd CreditCard-Transaction-Anomaly-Detection-Using-IsolationForest
   ```

2. Install the required libraries using the environment.yml file using conda:
   ```bash
   conda env create -f environment.yml
   ```
3. activate conda environment
   ```bash
   conda activate anomaly-detection
   ```
4. Dataset:
   get the dataset using the following link https://www.kaggle.com/datasets/ybifoundation/credit-card-transaction

5. Run the notebooks:
   
   Run the notebooks using the conda environment.

6. Explore the results:
   
   - The code performs exploratory data analysis, preprocessing, applies Isolation Forest for anomaly detection, and visualizes anomalies detected.

## Folder Structure

The project folder structure is organized as follows:
```
CreditCard-Transaction-Anomaly-Detection-Using-IsolationForest/
├── src
│ ├── utils.py
│ ├── anomaly_detection.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ └── anomaly_detection.ipynb
├── LICENSE
├── environment.yml
└── README.md
```

## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `anomaly_detection.py` : Functions for anomaly detection.py.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `anomaly_detection.py`: Notebook for anomaly detection.


## Results

The following is the visulized result for 0.001 of data.

![results](https://github.com/user-attachments/assets/1c5304f2-4632-4ae3-8678-b0623eb390c8)


## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Acknowledgments

- Inspired by tutorials and examples on anomaly detection and Isolation Forest.

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

