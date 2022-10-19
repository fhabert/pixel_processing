import numpy as np
import math
import pandas as pd

class Camera_Calibration(object):
    def __init__(self, df_points, df_worlds, nb_image):
        self.nb_image = nb_image
        self.df = df_points
        self.df_worlds = df_worlds
        self.images_points = {}
        self.world_points = []
        self.stored_phi_mat = {}
        self.stored_h_mat = {}
        self.K = None
        self.extrinsics_parameters = {}
        self.initiate_points()
        pass

    def initiate_points(self):
        images_titles = self.df.columns.tolist()[:self.nb_image*2]
        images_names = [f"img{i}" for i in range(1, self.nb_image+1)]
        images_list = []
        for item in images_titles:
            images_list.append(self.df[item])
        count = 0
        for i in range(0, len(images_list)-1, 2):
            result = zip(images_list[i], images_list[i+1])
            self.images_points[images_names[count]] = list(result)
            count +=1
        self.world_points = list(zip(self.df_worlds["worldPoints_x"], self.df_worlds["worldPoints_y"]))
        pass

    def h_matrix(self):
        # Create an array of arrays that be will be saving the coefficients of my h vector's variables
        H = [[], [], []]
        images_names = [f"img{i}" for i in range(1, self.nb_image+1)]
        # Itarating over each array of pixels of the images
        count = 0
        for _, points in self.images_points.items():
            linear_in_params = []
            for i in range(len(points)): 
                # Creating th two vectors that will be mutltiply to the h vector
                vect_img = [points[i][0], points[i][1], 1]
                vect_world = [self.world_points[i][0], self.world_points[i][1], 1]
                # Get the multiplication of my h vector and the world coordinate
                for k in range(len(H)):
                    for l in range(len(H)):
                        H[k].append(vect_world[l])
                # Get a cross product of my h vectors variable and the image points coordinate by saving the xi and yi vectors separately
                inner_H_yi = [0,0,0]
                for k in range(len(H)):
                    row3 = vect_img[2] * H[1][k]
                    inner_H_yi.append(row3)
                for k in range(len(H)):
                    row2 = -vect_img[1] * H[2][k]
                    inner_H_yi.append(row2)
                inner_H_xi = []
                for k in range(len(H)):
                    row3 = vect_img[2] * H[0][k]
                    inner_H_xi.append(row3)
                for _ in range(3):
                    inner_H_xi.append(0)
                for k in range(len(H)):
                    row1 = -vect_img[0] * H[2][k]
                    inner_H_xi.append(row1)
                linear_in_params.append(inner_H_yi)
                linear_in_params.append(inner_H_xi)
                # Reset the arrays to obtain new points h matrix
                inner_H_yi, inner_H_xi = [], []
                H = [[], [], []]
            self.stored_phi_mat[images_names[count]] = linear_in_params
            count += 1

        count = 0
        # print(self.stored_phi_mat["img1"])
        for _, value in self.stored_phi_mat.items():
            self.stored_h_mat[images_names[count]] = self.get_null_space(value)
            count += 1
        pass
    
    def b_matrix(self):
        all_images_v_vectors = []
        for _, value in self.stored_h_mat.items():
            h = np.array(value).reshape(3,3)
            vectors = { "v12": None, "v11": None, "v22": None }
            index_vectors = [[1,2], [1,1], [2,2]]
            counter = 0
            for key in vectors:
                i, j = index_vectors[counter][0], index_vectors[counter][1]
                vectors[key] = [ h[0][i]*h[0][j], (h[1][i]*h[0][j] + h[0][i]*h[1][j]),
                                 h[1][i]*h[1][j], (h[2][i]*h[0][j] + h[0][i]*h[2][j]), 
                                 (h[2][i]*h[1][j] + h[1][i]*h[2][j]), h[2][i]*h[2][j] ]
                counter += 1
            # combined_matrix = np.array([vectors["v12"], np.subtract(vectors["v11"], vectors["v22"])])
            all_images_v_vectors.append(vectors["v12"])
            all_images_v_vectors.append(np.subtract(vectors["v11"], vectors["v22"]))
        b_vector = self.get_null_space(all_images_v_vectors)
        return b_vector

    def get_null_space(self, matrix):
        _, S, V = np.linalg.svd(matrix)
        smallest_value = math.inf
        index_s_value = 0
        for i in range(len(S)):
            if S[i] < smallest_value:
                smallest_value = S[i]
                index_s_value = i
        vector = V[index_s_value]
        return vector
    
    def intrinsic_params(self, b_vector):
        b11, b12, b22, b13, b23, b33 = b_vector
        y = (b12*b13-b11*b23)/(b11*b22-b12**2)
        lambda_val = b33 - (b13**2+y*(b12*b13-b11*b23))/b11
        alpha_val = math.sqrt(lambda_val/b11)
        beta_val = math.sqrt((lambda_val*b11)/abs(b11*b22-b12**2))
        gamma_val = -(b12*(alpha_val**2)*beta_val)/lambda_val
        x = (gamma_val*y)/alpha_val - (b13*(alpha_val**2))/lambda_val
        np.set_printoptions(suppress=True)
        return np.around(np.array([alpha_val, gamma_val, x, 0, beta_val, y, 0, 0, lambda_val]), decimals=2)

    def reconstruct_intrinsic_mat(self,vect):
        vect = [[K[0], K[1], K[3]], [0, K[2],  K[4]], [0, 0, K[5]]]
        self.K = np.array(vect).reshape((3,3))
        return self.K

    def extrinsics_params(self):
        for key, value in self.stored_h_mat.items():
            K_inverse = np.linalg.inv(self.K)
            scale = 1/np.linalg.norm(K_inverse * np.array(value).T[0])
            r1 = scale * K_inverse * np.array(value).T[0]
            r2 = scale * K_inverse * np.array(value).T[1]
            r3 = r1 * r2
            P_org = scale * K_inverse * np.array(value).T[2]
            np.set_printoptions(suppress=True)
            self.extrinsics_parameters[key] = np.around(np.array([r1, r2, r3, P_org]).astype(np.float16), decimals=2)
        pass

# Fetching the images data points from the matlab file tored in a csv file
data = pd.read_csv("./datasets/image_points_1.csv", sep=";", encoding="utf_8")
data_world = pd.read_csv("./datasets/world_points_1.csv", sep=";", encoding="utf_8")
df = pd.DataFrame(data)
df_world = pd.DataFrame(data_world)

nb_image = 6
camera = Camera_Calibration(df, df_world, nb_image)
camera.h_matrix()
# print(camera.stored_h_mat)
b_vector = camera.b_matrix()
K = camera.intrinsic_params(b_vector)
K_matrix = camera.reconstruct_intrinsic_mat(K)
print(K_matrix)
E_params = camera.extrinsics_params()
# print(camera.extrinsics_parameters['img1'])

# test = [[0, 0, 0, 22, 22, 1, 27.1552, -4927.1552, -223.9616], [22, 22, 1, 0, 0, 0, 78.7999999, 78.7999, -335.4],\
#      [0, 0, 0, 22, 44, 1, -5.051, 70.102, -249.3205], [22, 44, 1, 0, 0, 0, 61.4666, 922.9332, -316.4303], \
#         [0, 0, 0, 22, 66, 1, 25.430399999, -18076.2912, -273.8832], [22, 66, 1, 0, 0, 0, 565.87000001, -19667.6002, -297.9945], \
#             [0, 0, 0, 22, 88, 1, 44.0408, 6176.1632, -297.4564], [22, 88, 1, 0, 0, 0, 64.08001, 656.3004, -280.1856],\
#                  [0, 0, 0, 22, 110, 1, 53.7899995, -35268.904, -320.6264], [22, 110, 1, 0, 0, 0, -5781.4328, -28907.7, -262.7924],\
#                      [0, 0, 0, 44, 0, 1, 492.9208, -0.0, 15.7482], [44, 0, 1, 0, 0, 0, 882.902, -0.0, -383.7033], \
#                         [0, 0, 0, 44, 22, 1, 620.588, -5310.294, -241.377], [44, 22, 1, 0, 0, 0,909.53000, -8004.7005, -363.853], \
#      [0, 0, 0, 44, 66, 1, 767.2336, 40.8504, -0.1644], [44, 66, 1, 0, 0, 0, -14338.5, -21507.75, -325.875]]

# print(camera.get_null_space(test)),
matlab_test = np.array([818.209914441912,0,297.795351588878,0,818.280573772713,212.136282255704,0,0,1]).reshape((3,3))