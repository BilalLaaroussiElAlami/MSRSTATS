import matplotlib.pyplot as plt

def indexCategoryValue(arr,val):
    for i in range(len(arr)):
        if arr[i] > val:
            return i-1
    return -1


def preProcess(file_path):
    binSize = 20
    categories = list(range(100,300,binSize))
    trueValuesArchitecure1  = [0] * len(categories)
    falseValuesArchitecure1 = [0] * len(categories)
    trueValuesArchitecure2  = [0] * len(categories)
    falseValuesArchitecure2 = [0] * len(categories)


    # Read the data from the text file, count the number of true and false values for each category
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        data = line.strip().split('\t')
        architecture = data[0].strip("\"")
        complexity = int(data[1])
        value = data[2]
        print(list(architecture))
        if architecture == "architecture 1":
            print("ok")
            if value == 'TRUE':
                trueValuesArchitecure1[indexCategoryValue(categories,complexity)] += 1
            else:
                falseValuesArchitecure1[indexCategoryValue(categories,complexity)] += 1
        else:
            if value == 'TRUE':
                trueValuesArchitecure2[indexCategoryValue(categories,complexity)] += 1
            else:
                falseValuesArchitecure2[indexCategoryValue(categories,complexity)] += 1
    print(categories)
    print(trueValuesArchitecure1)
    print(falseValuesArchitecure1)
    print(trueValuesArchitecure2)
    print(falseValuesArchitecure2)

    str_array = [str(num) for num in categories]

    plt.bar(str_array, trueValuesArchitecure1, color='r')
    plt.bar(str_array, falseValuesArchitecure1, bottom=trueValuesArchitecure1, color='b')
    plt.title("Architecture 1 Stacked Barchart")
    plt.show()

    plt.bar(str_array, trueValuesArchitecure2, color='r')
    plt.bar(str_array, falseValuesArchitecure2, bottom=trueValuesArchitecure2, color='b')
    plt.title("Architecture 2 Stacked Barchart")
    plt.show()


    ratioTrueValuesArchitecure1 = list(map(lambda x,y: x/ (x+y), trueValuesArchitecure1,falseValuesArchitecure1))
    plt.bar(str_array, ratioTrueValuesArchitecure1, color='b')
    plt.title("Architecture 1 Stacked Barchart")
    plt.ylim(0, 1)
    plt.show()

    ratioTrueValuesArchitecure2 = list(map(lambda x, y: x / (x + y), trueValuesArchitecure2, falseValuesArchitecure2))
    plt.bar(str_array, ratioTrueValuesArchitecure2, color='b')
    plt.title("Architecture 2 Stacked Barchart")
    plt.ylim(0, 1)
    plt.show()

def bar_plot():
    # create data
    x = ['A', 'B', 'C', 'D']
    y1 = [10, 20, 10, 30]
    y2 = [20, 25, 15, 25]

    # plot bars in stack manner
    plt.bar(x, y1, color='r')
    plt.bar(x, y2, bottom=y1, color='b')
    plt.show()

if __name__ == '__main__':
    #bar_plot()
    print(indexCategoryValue([1,20,30,40], 19))
    preProcess('StatisticsMWO/Question2_3.txt')
