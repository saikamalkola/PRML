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

trainingSet_New = trainingSet
for x in range(len(trainingSet)):
    trainingSet_New[x].append(training_labels[x])

trainingSet_N2 = testSet
for x in range(len(testSet)):
    trainingSet_N2[x].append(test_labels[x])

with open('data_percp.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(trainingSet_N2)
    writer.writerows(trainingSet_New)
