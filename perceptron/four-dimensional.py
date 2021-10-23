import numpy as np
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d

class Perceptron:
    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 5000

    def read_file(slef, str):
        list_data = []
        list_label = []
        train_file = open(str)
        for line in train_file:
            line = line.split(',')
            for i in range(len(line) - 1):
                line[i] = float(line[i])
            list_data.append(line[0:4])
            label = line[-1]
            if label == "Iris-setosa\n":
                list_label.append(1)
            else:
                list_label.append(-1)
        train_file.close()

        return list_data, list_label

    def train(self, list_data, list_label):
        global w, b

        data = np.array(list_data)
        label = np.array(list_label)
        w = np.array([0, 0, 0, 0])
        b = 0

        f = (np.dot(data, w.T) + b) * label
        idx = np.where(f <= 0)
        i = 1
        while f[idx].size != 0 and i <self.max_iteration:
            point = np.random.randint((f[idx].shape[0]))
            x = data[idx[0][point], :]
            y = label[idx[0][point]]
            w = w + self.learning_step * x * y
            b = b + self.learning_step * y
            print('Iteration:%d  w:%s  b:%s' % (i, w, b))
            i += 1
            f = (np.dot(data, w.T) + b) * label
            idx = np.where(f <= 0)

        return w, b

    def train2(self, list_data, list_label):
        global w, b

        data = np.array(list_data)
        label = np.array(list_label)
        w = np.array([0, 0, 0, 0])
        b = 0
        ni = [0]*len(list_data)

        f = (np.dot(data, w.T) + b) * label
        idx = np.where(f <= 0)
        i = 1
        while f[idx].size != 0 and i <self.max_iteration:
            point = np.random.randint((f[idx].shape[0]))
            x = data[idx[0][point], :]
            y = label[idx[0][point]]
            ni[point] += 1
            w = self.learning_step * ni[point] * x * y + w
            b = self.learning_step * ni[point] * y + b
            print('Iteration:%d  w:%s  b:%s' % (i, w, b))
            i += 1
            f = (np.dot(data, w.T) + b) * label
            idx = np.where(f <= 0)

        return w, b

    
    def predict(self, list_data):
        fainl_label = []
        for [x, y, z, t] in list_data:
            data = np.array([x, y, z, t])
            f = np.dot(data, w.T) + b
            if f > 0:
                fainl_label.append(1)
            else:
                fainl_label.append(-1)
        return fainl_label


if __name__ == "__main__":
    time1 = time.time()

    p = Perceptron()
    list_data, list_label = p.read_file("train.data")
    print("开始训练模型")
    w, b = p.train2(list_data, list_label)
    time2 = time.time()

    print("训练用时：", time2 -time1, "second", "\n")
    print("参数为：", w, b, "\n")

    time3 = time.time()
    print("开始预测")
    list_data, list_label = p.read_file("test.data")
    fainl_label = p.predict(list_data)
    score = accuracy_score(list_label, fainl_label)

    print("预测用时：", time3 -time2, "second", "\n")
    print("The accruacy socre is ", score)


