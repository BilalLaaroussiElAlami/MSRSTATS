import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import pearsonr


#--------------start question 1-----------------

def draw_boxplots(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')
    # Generate boxplots for each column
    plt.figure(figsize=(8, 6))
    data.boxplot(notch = True)
    plt.xlabel('Inteface')
    plt.ylabel('Response Time')
    plt.show()


def create_qq_plot(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')
    # Create Q-Q plot for each column
    plt.figure(figsize=(8, 6))
    for column in data.columns:
        sm.qqplot(data[column], line ='45')
        plt.title(f'Q-Q Plot for {column}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.show()


#make a quantile-quantile plot for the data in the file at the given path
def qq_plot(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')
    # Create Q-Q plot for each column
    plt.figure(figsize=(8, 6))
    for column in data.columns[:1]:
        stats.probplot(data[column], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {column}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.show()


def create_histogram(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')

    # Create histograms for each column
    plt.figure(figsize=(8, 6))

    for column in data.columns:
        plt.hist(data[column], bins=10, alpha=0.7, label=column)

    plt.title('Histogram for Each Column')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def t_test():
    # Example data (replace these with your actual response time data)
    web_form_1_times = [34.37,29.13,26.9,30.1,24.68,40.37,15.1,33.62,32.71,34.63,30,34.11,27.62,32.71,38.05,40.45,38.95,29.13,24.18,16.1]
    web_form_2_times = [37.1,30,20.5,29.8,22.4,34.3,17.1,34.9,32,31.1,34.2,53.5,23.2,32.1,53.5,45.5,32.5,31.4,39.2,33.5]
    # Perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(web_form_1_times, web_form_2_times)
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
 #----------------end question 1-----------------


#----------------start question 2-----------------
#make a function that takes as input a file path. The file contains two columns of data, one for each group. make a scatter plot
#of the data in the file, with the first column on the x-axis and the second column on the y-axis.
def scatter_plot(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:,0], data.iloc[:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def correlation():
# Data
    realism = [1, 3, 6, 6, 6, 8, 10, 10, 11, 12, 14, 14, 15, 15, 17, 19, 20, 20, 21, 22, 26,
              28, 30, 30, 31, 32, 32, 34, 35, 35, 36, 36, 36, 37, 37, 37, 39, 39, 39, 41, 42,
              42, 43, 45, 45, 49, 50, 51, 51, 52, 52, 53, 53, 54, 56, 57, 59, 59, 60, 60, 62,
              62, 63, 63, 67, 67, 68, 69, 69, 71, 75, 75, 77, 77, 78, 79, 79, 79, 80, 81, 82,
              83, 83, 84, 85, 86, 86, 87, 90, 91, 92, 93, 93, 93, 94, 94, 95, 97, 97, 98]

    acceptability = [5, 14, 23, 19, 24, 30, 34, 37, 38, 41, 49, 50, 51, 54, 56, 59, 64, 61, 66,
                72, 73, 80, 82, 84, 88, 87, 90, 88, 89, 90, 92, 90, 91, 91, 95, 95, 95, 98,
                100, 99, 98, 97, 98, 97, 97, 99, 100, 100, 99, 100, 100, 100, 94, 96, 100,
                96, 99, 95, 94, 95, 89, 88, 89, 90, 88, 86, 84, 75, 72, 71, 71, 70, 66, 67,
                70, 61, 62, 58, 56, 53, 52, 51, 46, 51, 45, 30, 31, 34, 30, 28, 26, 24, 21,
                16, 7, 7, 6]

# Calculate Pearson correlation coefficient and p-value
    corr, p_value = pearsonr(realism, acceptability)
    print(f"Pearson correlation coefficient: {corr}")
    print(f"P-value: {p_value}")

def correlation2(file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
         lines = file.readlines()

    # Extract data into lists
    realism = []
    acceptability = []
    for line in lines[1:]:
        data = line.strip().split('\t')
        realism.append(int(data[0]))
        acceptability.append(int(data[1]))

    # Calculate Pearson correlation coefficient and p-value
    corr, p_value = pearsonr(realism, acceptability)
    print(f"Pearson correlation coefficient: {corr}")
    print(f"P-value: {p_value}")



#----------------end question 2-----------------


if __name__ == '__main__':
    """
    file_path = 'StatisticsMWO/Question1_2.txt'
    draw_boxplots(file_path)
    create_qq_plot(file_path)
    create_histogram(file_path)
    qq_plot(file_path)
    t_test()
    """
    file_path_question_3 = 'StatisticsMWO/Question3_3.txt'
    scatter_plot(file_path_question_3)
    correlation2(file_path_question_3)