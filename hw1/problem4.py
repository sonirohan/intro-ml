"""Note that this problem uses the dataset from the following link:
https://www.kaggle.com/datasets/maajdl/yeh-concret-data

We also assume that we've already unzipped the dataset.
Now, it's located at hw1/data/question_4_data.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():

    # Load the dataset
    data = pd.read_csv('hw1/data/question_4_data.csv')

    print("Shape of the dataset:", data.shape)
    print("----")
    print("Datatypes: \n", data.dtypes)
    print("----")
    print(data.describe())
    print("----")
    print("Number of samples: " + str(len(data)))
    print("Number of features: " + str(len(data.columns) - 1))  # Exclude the target variable
    print("Are there missing values? " + str(data.isnull().values.any()))


    """    
    plt.figure(figsize=(10, 6))
    plt.title('Distribution of Concrete Compressive Strength')
    plt.xlabel('Concrete Compressive Strength')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('concrete_compressive_strength_distribution.png')
    """

if __name__ == "__main__":    
    main()