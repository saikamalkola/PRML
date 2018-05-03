import csv
import random
import math

def normal(x, mean, variance):
    pi = math.pi
    val = 1/((2 * pi * variance) ** 0.5)
    val *= (math.exp(-1*((x - mean)**2) / (2*variance)))
    return val

def find_mean_variance(array):
    mean = []
    variance = []
    for x in range(len(array[0])):
        n = len(array)
        summation = 0
        for y in range(n):
            summation += array[y][x]
        summation = summation / n
        mean.append(summation)
    for x in range(len(array[0])):
        n = len(array)
        summation = 0
        for y in range(n):
            summation += ((array[y][x] - mean[x]) * (array[y][x] - mean[x]))
        summation = summation / n
        variance.append(summation)
    return mean, variance

def cal_joint_probs(test, mean, variance, n):
    P = 1
    for x in range(n):
        P *= normal(test[x], mean[x], variance[x])
    return P

def cal_probs(test,mean_M, var_M,mean_B, var_B,n):
    P1 = cal_joint_probs(test, mean_M, var_M, n)
    P2 = cal_joint_probs(test, mean_B, var_B, n)
    return P1,P2

def predict(P1, P2):
    if(P1 >= P2):
        prediction = 'M'
    else:
        prediction = 'B'
    return prediction

def loss(x,y):
    if(x == y):
        return 0
    else:
        return 1

def test_vector(test, mean_M, var_M,mean_B, var_B,n):
    P1,P2 = cal_probs(test, mean_M, var_M,mean_B, var_B,n)
    prediction = predict(P1, P2)
    return prediction

def score(array, array_labels, mean_M, var_M,mean_B, var_B):
    n_test = len(array)
    n = len(array[0])
    loss_count = 0
    for x in range(n_test):
        test = array[x]
        test_label = array_labels[x]
        P1,P2 = cal_probs(test, mean_M, var_M,mean_B, var_B,n)
        prediction = predict(P1, P2)
        loss_count += (1 - loss(prediction, test_label))

    return (loss_count / n_test)

trainingSet=[]
testSet=[]
features = []
training_labels = []
test_labels = []

with open('data.csv', 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    features = dataset.pop(0)
    #print(features)
    for x in range(1,len(dataset)-1):
        for y in range(2, len(features) - 1):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < 0.66:
            trainingSet.append(dataset[x])
        else:
	        testSet.append(dataset[x])

for x in range(len(trainingSet)):
    trainingSet[x].pop(0)
    training_labels.append(trainingSet[x].pop(0))
for x in range(len(testSet)):
    testSet[x].pop(0)
    test_labels.append(testSet[x].pop(0))

#Maximum Likelihood Estimation Classifier - Assuming denisties are Normal
#normalise training data and testing dataset
min = [0 for row in range(len(trainingSet[0]))]
max = [0 for row in range(len(trainingSet[0]))]

for x in range(len(trainingSet[0])):
    min[x] = trainingSet[0][x]
    max[x] = trainingSet[0][x]
    for y in range(len(trainingSet)):
        if(trainingSet[y][x] < min[x]):
            min[x] = trainingSet[y][x]
        if(trainingSet[y][x] > max[x]):
            max[x] = trainingSet[y][x]
for x in range(len(testSet[0])):
    for y in range(len(testSet)):
        if(testSet[y][x] < min[x]):
            min[x] = testSet[y][x]
        if(trainingSet[y][x] > max[x]):
            max[x] = testSet[y][x]

for x in range(len(trainingSet[0])):
    for y in range(len(trainingSet)):
        trainingSet[y][x] = (trainingSet[y][x] - min[x]) / (max[x] - min[x])
for x in range(len(testSet[0])):
    for y in range(len(testSet)):
        testSet[y][x] = (testSet[y][x] - min[x]) / (max[x] - min[x])
# Finding Mean and Standard deviation of data
trainingSet_M = []
trainingSet_B = []

for x in range(len(trainingSet)):
    if(training_labels[x] == 'M'):
        trainingSet_M.append(trainingSet[x])
    elif(training_labels[x] == 'B'):
        trainingSet_B.append(trainingSet[x])

mean_M, var_M = find_mean_variance(trainingSet_M)
mean_B, var_B = find_mean_variance(trainingSet_B)

train_score = score(trainingSet, training_labels, mean_M, var_M,mean_B, var_B)
test_score = score(testSet, test_labels, mean_M, var_M,mean_B, var_B)

print("ML Estimation with Split ratio of data = 0.67")
print("Train Score: ", end = " ")
print(train_score * 100, end = "% ")
print("Test Score: ", end = " ")
print(test_score * 100,end = "% ")
