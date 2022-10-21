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
        EX2, EX3, EX4 = np.ones(shape=(1,3)), np.ones(shape=(1,3)), np.ones(shape=(1,3))
        wp = self.world_points
        container = np.ones(shape=(3, 3))
        indexes = [(0, 0), (1, 0), 0, (0, 1), (1,1), 1, 0, 1, len(wp)]
        xi = []
        yi = []
        for k in range(len(wp)):
            for l in range(len(wp[0])):
                if l == 0:
                    xi.append(wp[k][l])
                elif l == 1:
                    yi.append(wp[k][l])
        counter = 0
        values = [xi, yi]
        for i in range(len(container)):
            for j in range(len(container[0])):
                if type(indexes[counter]) is tuple:
                    final_sum = np.sum(np.array([values[indexes[counter][0]][m] * values[indexes[counter][1]][m] for m in range(len(xi)-1)]))
                elif indexes[counter] != 0 and indexes[counter] != 1:
                    final_sum = indexes[counter]
                else:
                    final_sum = np.sum(np.array(values[indexes[counter]]))
                container[i][j] = final_sum
                counter += 1
        np.set_printoptions(suppress=True)
        print(container)
        inverse_container = np.linalg.inv(container)
        print(inverse_container)
        pass


world_points_robot = [[-22,-22, 0, 1], [0, 22, 0,1], [22, 66, 0, 1], [44,22, 0,1], [66,88,0,1], [154,-22,0,1]]
tcp_points = [[-121.6,222.4, -20],[-100.4, 178, -20.04], [-79.16, 133.4, -20.04], [-56.39, 177.2, -19.69], [-35.55, 110.9, -20.04], [54.32, 219.3,19.7]]
robot = Robot_calibration(world_points_robot,tcp_points)
positions = robot.position()