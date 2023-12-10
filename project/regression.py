import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def logistic_regression(file_path):
    # Assuming your data is in a CSV file named 'data.csv'
    data = pd.read_csv(file_path , delimiter='\t')

    X = data[['Complexity']]  # Independent variable
    y = data['Convergence']  # Dependent variable

    model = LogisticRegression()
    model.fit(X, y)

    # Predict probabilities for the complexity values in your dataset
    probabilities = model.predict_proba(X)[:, 1]  # Probabilities of the positive class

    # Find the complexity value where the predicted probability converges to 50%
    convergence_value = X[np.abs(probabilities - 0.5).argmin()]['complexity']

    # Plotting the data and logistic regression curve
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, probabilities, color='red', label='Logistic Regression Curve')

    # Highlight where 50% convergence occurs
    plt.axvline(x=convergence_value, color='green', linestyle='--', label='50% Convergence')

    plt.xlabel('Complexity')
    plt.ylabel('Output')
    plt.title('Logistic Regression')
    plt.legend()
    plt.show()
    print(f"The value where 50% convergence occurs is: {convergence_value}")

if __name__ == '__main__':
    file_path = 'StatisticsMWO/Question2_3.txt'
    logistic_regression(file_path)
