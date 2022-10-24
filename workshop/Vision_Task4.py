import math
import numpy as np
import json

name_files = ["Tcw", "Trw", "K",  "PObject"]
values = []

# Fetching the stored values for the different parameters 
for item in name_files:
    with open(f"./{item}.json") as f:
        data = json.load(f)
    key = f"{item}_matrix"
    values.append(data[key])
f.close()
Tcw, Trw, K, P_object = values

np.set_printoptions(suppress=True)
Tcw_without_col3 = []
Tcw_reshape = np.array(Tcw).reshape(3,4)
# Removing the r3 column from the Tcw matrix to allow matrix multiplication
# with tthe K matrix
for i in range(len(Tcw_reshape)):
    inner_list = []
    for j in range(len(Tcw_reshape[0])):
        if j != 2:
            inner_list.append(Tcw_reshape[i][j])
    Tcw_without_col3.append(inner_list)

# Finding the H matrix based on the K matrix and Tcw
H = np.matmul(np.array(K).reshape(3,3), np.array(Tcw_without_col3).reshape(3,3))
# Obtaining the world coordinate
combining = np.around(np.matmul(np.linalg.inv(H), np.array(P_object).T), decimals=4)
combining_add_one = np.array([combining[0], combining[1], combining[2], 1])
# Finding the coordinate of the robot with the world coordinate
robot_coordinate = np.matmul(np.array(Trw).reshape(4,4), combining_add_one)
print("These are the coordinates to which the robot can move to: \n", robot_coordinate)