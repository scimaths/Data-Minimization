import os
import sys
import numpy as np
import json
# read all files from datapoints folder
files = os.listdir("datapoints")
files.sort()

data_1a = []
data_1b = []
data_2a = []
data_2b = []
# read all files from files variable and parse the file name
for file in files:
    file_name = file.split("-")
    print(file_name)
    
    if file_name[1] == "1" and file_name[3] == "True" and file_name[5] == "60" and file_name[7] == "0.6" and file_name[9] == "5" and file_name[13] == "500.json":
        with open("datapoints/" + file, "rb") as f:
            data_1a.append(int(file_name[11]))
            lines = f.readlines()
            data_1b.append(json.loads(lines[0])["Error"]/len(json.loads(lines[0])["Actual"]))
    
    if file_name[1] == "2" and file_name[3] == "True" and file_name[5] == "60" and file_name[7] == "0.6" and file_name[9] == "5" and file_name[13] == "500.json":
        with open("datapoints/" + file, "rb") as f:
            data_2a.append(int(file_name[11]))
            lines = f.readlines()
            data_2b.append(json.loads(lines[0])["Error"]/len(json.loads(lines[0])["Actual"]))

print(data_1b, data_2b)
import matplotlib.pyplot as plt
plt.scatter(data_1a, data_1b, label="w/o Data Minimization")
plt.scatter(data_2a, data_2b, label="with Data Minimization")
plt.xlabel("Number of time slots used to train")
plt.ylabel("Error")
plt.legend()
# plt.show()
plt.savefig("plot.png")
