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



def t_test():
    # Example data (replace these with your actual response time data)
    web_form_1_times = [34.37,29.13,26.9,30.1,24.68,40.37,15.1,33.62,32.71,34.63,30,34.11,27.62,32.71,38.05,40.45,38.95,29.13,24.18,16.1]
    web_form_2_times = [37.1,30,20.5,29.8,22.4,34.3,17.1,34.9,32,31.1,34.2,53.5,23.2,32.1,53.5,45.5,32.5,31.4,39.2,33.5]
    # Perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(web_form_1_times, web_form_2_times)
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
 #----------------end question 1-----------------


#----------------start question 3-----------------
#make a function that takes as input a file path. The file contains two columns of data, one for each group. make a scatter plot
#of the data in the file, with the first column on the x-axis and the second column on the y-axis.
def scatter_plot(file_path):
    # Read the data from the text file
    data = pd.read_csv(file_path, delimiter='\t')
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:,0], data.iloc[:,1])
    plt.xlabel('realism')
    plt.ylabel('acceptability')
    plt.show()



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

#----------------end question 3-----------------




#----------------start question 4-----------------
#R CODE FOR DETERMING SAMPLE SIZE
# pwr.r.test(r = 0.5, sig.level = 0.05, power = 0.8)

def type_1_error():
    # Parameters
    num_simulations = 1000
    sample_size = 29
    effect_size = 0.5
    power = 0.8
    significance = 0.05

    # Initialize variables to count significant differences
    significant_count = 0
    for _ in range(num_simulations):
        # Generate two sets of uniformly distributed samples
        sample1 = np.random.uniform(0, 1, sample_size)
        sample2 = np.random.uniform(0, 1, sample_size)

        # Calculate mean and standard deviation for each sample
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((sample_size - 1) * std1 ** 2 + (sample_size - 1) * std2 ** 2) / (2 * (sample_size - 1)))

        # Calculate t-statistic
        t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(2 / sample_size))

        # Calculate degrees of freedom
        df = 2 * sample_size - 2

        # Calculate critical t-value
        critical_t = stats.t.ppf(1 - significance / 2, df)

        # Check for significance
        if abs(t_stat) > critical_t:
            significant_count += 1

    print("significant count: ", significant_count)
    # Calculate Type I error rate
    type_I_error_rate = significant_count / num_simulations
    print(f"Type I error rate: {type_I_error_rate}")


def power():
    import numpy as np
    from scipy import stats

    # Function to generate uniformly distributed samples
    def generate_uniform_samples(size, a, b):
        return np.random.uniform(a, b, size)

    # Parameters
    sample_size = 100  # Size of each sample
    effect_size = 0.5  # Desired effect size
    num_samples = 1000  # Number of pairs of samples

    significant_count = 0  # Counter for significant differences

    # Generate pairs of samples and perform t-test
    for _ in range(num_samples):
        # Generate two sets of samples
        sample1 = generate_uniform_samples(sample_size, 0, 1)
        sample2 = generate_uniform_samples(sample_size, effect_size, 1 + effect_size)

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(sample1, sample2)

        # Check for significance (assuming alpha = 0.05)
        if p_value < 0.05:
            significant_count += 1

    # Calculate proportion of significant differences
    proportion_significant = significant_count / num_samples
    print(f"Proportion of significant differences: {proportion_significant}")


if __name__ == '__main__':
    #Question 1
    file_path = 'StatisticsMWO/Question1_2.txt'
    draw_boxplots(file_path)
    create_qq_plot(file_path)
    t_test()

    #Question 3
    file_path_question_3 = 'StatisticsMWO/Question3_3.txt'
    scatter_plot(file_path_question_3)
    correlation2(file_path_question_3)


    #question 4
    type_1_error()
    power()