from io import FileIO
from os import name
import sys
import random

if __name__ == "__main__":
    object = open("iris.data")
    list = []
    while 1:
        line = object.readline()
        if not line:
            break
        if ('Iris-setosa' in line or 'Iris-versicolor' in line):
            list.append(line)
    
    random.shuffle(list)
    sublist1 = list[:30]
    sublist2 = list[30:]

    object.close()

    subfile = open("train.data", mode='w+')
    it = iter(sublist2)
    for x in it:
        subfile.write(x)
    
    subfile.close()

    subfile = open("test.data", mode='w+')
    it = iter(sublist1)
    for x in it:
        subfile.write(x)
    
    subfile.close()