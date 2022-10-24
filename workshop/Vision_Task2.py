import numpy as np
import json

class Robot_calibration(object):
    def __init__(self, world_p, tcp_p):
        self.world_points = world_p
        self.tcp = tcp_p
        self.unknown_params = np.ones(shape=(4,4))
        pass

    def position(self):
        wp = np.array(self.world_points).reshape(6,4)
        tcp = np.array(self.tcp).reshape(6,4)
        rows = { "row1": None, "row2": None, "row3": None }
        n = len(wp)
        container = np.array([  np.sum(np.matmul(wp[:, 0], wp[:, 0])), \
                    np.sum(np.matmul(wp[:, 1], wp[:, 0])),\
                    np.sum(wp[:, 0]), np.sum(np.matmul(wp[:, 0], wp[:, 1])), \
                    np.sum(np.matmul(wp[:, 1], wp[:, 1])), np.sum(wp[:, 1]),\
                    np.sum(wp[:, 0]), np.sum(wp[:, 1]), n \
                  ]).reshape(3,3)
        inverse_container = np.linalg.inv(container)
        i = 0
        for key in rows:
            right_mat = np.array([ np.sum(np.matmul(tcp[:, i], wp[:, 0])), \
                              np.sum(np.matmul(tcp[:, i], wp[:, 1])), \
                              np.sum(tcp[:, i]) \
                            ])
            value = np.matmul(inverse_container, right_mat)
            rows[key] = value
            i += 1
        
        col3 = np.cross(np.array([rows["row1"][0], rows["row2"][0],rows["row3"][0]]), \
                        np.array([rows["row1"][1], rows["row2"][1],rows["row3"][1]]))
        recombining = []
        count = 0
        for _, value in rows.items():
            temp = list(value)
            temp.insert(2, col3[count])
            recombining.append(temp)
            count += 1
        recombining.append([0, 0, 0, 1])
        return recombining


world_points_robot = [[-22,-22, 0, 1], [0, 22, 0,1], [22, 66, 0, 1], \
                      [44,22, 0,1], [66,88,0,1], [154,-22,0,1]]
tcp_points = [[-121.6, 222.4, -20, 1], [-100.4, 178, -20.04, 1], \
              [-79.16, 133.4, -20.04, 1], [-56.39, 177.2, -19.69, 1], \
              [-35.55, 110.9, -20.04, 1], [54.32, 219.3,19.7, 1]]
robot = Robot_calibration(world_points_robot,tcp_points)
positions = robot.position()
save_Trw = json.dumps({ "Trw_matrix": list(np.array(positions).flatten())})
with open("./Trw.json", "w") as K_file:
    K_file.write(f"{save_Trw}")
K_file.close()