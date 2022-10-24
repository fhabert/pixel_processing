import math
import matplotlib.pyplot as plt
import numpy as np

class Robot_calibration(object):
    def __init__(self, world_p, tcp_p):
        self.world_points = world_p
        self.tcp = tcp_p
        self.unknown_params = np.ones(shape=(4,4))
        pass

    def position(self):
        wp = np.array(self.world_points).reshape(6,4)
        print(wp[:, 0])
        row1, row2, row3 = np.ones(3), np.ones(3), np.ones(3)
        x_matrix = np.ones(shape=(3, 1))
        n = len(wp)
        container = np.array([  np.sum(np.matmul(wp[:, 0], wp[:, 0])), \
                    np.sum(np.matmul(wp[:, 1], wp[:, 0])),\
                    np.sum(wp[:, 0]), np.sum(np.matmul(wp[:, 0], wp[:, 1])), \
                    np.sum(np.matmul(wp[:, 1], wp[:, 1])), np.sum( wp[:, 1]),\
                    np.sum(wp[:, 0]), np.sum(wp[:, 1]), n \
                  ]).reshape(3,3)
        inverse_container = np.linalg.inv(container)
        count = 0

        # for i in range(3):
        #     for j in range(3):
        #         container[i][j] = indexes[count]
        # xi = []
        # yi = []
        # for k in range(len(wp)):
        #     for l in range(len(wp[0])):
        #         if l == 0:
        #             xi.append(wp[k][l])
        #         elif l == 1:
        #             yi.append(wp[k][l])
        # counter = 0
        # values = [xi, yi]
        # for i in range(len(container)):
        #     for j in range(len(container[0])):
        #         if type(indexes[counter]) is tuple:
        #             final_sum = np.sum(np.array([values[indexes[counter][0]][m] * values[indexes[counter][1]][m] for m in range(len(xi)-1)]))
        #         elif indexes[counter] != 0 and indexes[counter] != 1:
        #             final_sum = indexes[counter]
        #         else:
        #             final_sum = np.sum(np.array(values[indexes[counter]]))
        #         container[i][j] = final_sum
        #         counter += 1
        # np.set_printoptions(suppress=True)
        # print(container)
        # inverse_container = np.linalg.inv(container)
        # print(inverse_container)
        # col3 = np.cross(np.array([row1[0], row2[0], row3[0]]), np.array([row1[1], row2[1], row3[1]]))
        # print(col3)
        pass


world_points_robot = [[-22,-22, 0, 1], [0, 22, 0,1], [22, 66, 0, 1], \
                      [44,22, 0,1], [66,88,0,1], [154,-22,0,1]]
tcp_points = [[-121.6, 222.4, -20, 1], [-100.4, 178, -20.04, 1], \
              [-79.16, 133.4, -20.04, 1], [-56.39, 177.2, -19.69, 1], \
              [-35.55, 110.9, -20.04, 1], [54.32, 219.3,19.7, 1]]
robot = Robot_calibration(world_points_robot,tcp_points)
positions = robot.position()