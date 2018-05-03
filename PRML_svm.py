import csv
import random
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        print(len(all_data))
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1


        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]



        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()

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

#normalise training data and testing dataset
mini = [0 for row in range(len(trainingSet[0]))]
maxi = [0 for row in range(len(trainingSet[0]))]

for x in range(len(trainingSet[0])):
    mini[x] = trainingSet[0][x]
    maxi[x] = trainingSet[0][x]
    for y in range(len(trainingSet)):
        if(trainingSet[y][x] < mini[x]):
            mini[x] = trainingSet[y][x]
        if(trainingSet[y][x] > maxi[x]):
            maxi[x] = trainingSet[y][x]
for x in range(len(testSet[0])):
    for y in range(len(testSet)):
        if(testSet[y][x] < mini[x]):
            mini[x] = testSet[y][x]
        if(trainingSet[y][x] > maxi[x]):
            maxi[x] = testSet[y][x]

for x in range(len(trainingSet[0])):
    for y in range(len(trainingSet)):
        trainingSet[y][x] = (trainingSet[y][x] - mini[x]) / (maxi[x] - mini[x])
for x in range(len(testSet[0])):
    for y in range(len(testSet)):
        testSet[y][x] = (testSet[y][x] - mini[x]) / (maxi[x] - mini[x])
# Finding Mean and Standard deviation of data
trainingSet_M = []
trainingSet_B = []

for x in range(len(trainingSet)):
    if(training_labels[x] == 'M'):
        trainingSet_M.append(trainingSet[x])
    elif(training_labels[x] == 'B'):
        trainingSet_B.append(trainingSet[x])

data_dict = {-1:np.array(trainingSet_M),
             1:np.array(trainingSet_B)}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

for p in predict_us:
    svm.predict(p)

svm.visualize()
