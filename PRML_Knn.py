import csv
import random
import math
import operator

def euclideanDistance(X1, X2, length):
	distance = 0
	for x in range(length):
		distance += pow((X1[x] - X2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x],training_labels[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])
	return neighbors

def class_count(neighbors):
    count_B = 0
    count_M = 0
    for x in range(len(neighbors)):
        if(neighbors[x][1] == 'M'):
            count_M += 1
        elif(neighbors[x][1] == 'B'):
            count_B += 1
    return count_M, count_B

#def test_vector():

def score(array, array_labels, trainingSet, k):
    n = len(array)
    loss_count = 0
    for x in range(n):
        test = array[x];
        test_label = array_labels[x];
        neighbors = getNeighbors(trainingSet, test, k)
        M_count, B_count = class_count(neighbors)
        prediction = predict(M_count, B_count)
        loss_count += (1 - loss(prediction, test_label))
    return (loss_count / n)

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
        if random.random() < 0.67:
            trainingSet.append(dataset[x])
        else:
	        testSet.append(dataset[x])

for x in range(len(trainingSet)):
    trainingSet[x].pop(0)
    training_labels.append(trainingSet[x].pop(0))
for x in range(len(testSet)):
    testSet[x].pop(0)
    test_labels.append(testSet[x].pop(0))

# #normalise training data and testing dataset
# min = [0 for row in range(len(trainingSet[0]))]
# max = [0 for row in range(len(trainingSet[0]))]
#
# for x in range(len(trainingSet[0])):
#     min[x] = trainingSet[0][x]
#     max[x] = trainingSet[0][x]
#     for y in range(len(trainingSet)):
#         if(trainingSet[y][x] < min[x]):
#             min[x] = trainingSet[y][x]
#         if(trainingSet[y][x] > max[x]):
#             max[x] = trainingSet[y][x]
# for x in range(len(testSet[0])):
#     for y in range(len(testSet)):
#         if(testSet[y][x] < min[x]):
#             min[x] = testSet[y][x]
#         if(trainingSet[y][x] > max[x]):
#             max[x] = testSet[y][x]
#
# for x in range(len(trainingSet[0])):
#     for y in range(len(trainingSet)):
#         trainingSet[y][x] = (trainingSet[y][x] - min[x]) / (max[x] - min[x])
# for x in range(len(testSet[0])):
#     for y in range(len(testSet)):
#         testSet[y][x] = (testSet[y][x] - min[x]) / (max[x] - min[x])

k = 6
# test = testSet[20];
# test_label = test_labels[20];
# neighbors = getNeighbors(trainingSet, test, k)
# M_count, B_count = class_count(neighbors)
# prediction = predict(M_count, B_count)
# print(prediction)
print(score(trainingSet, training_labels, trainingSet, k))
