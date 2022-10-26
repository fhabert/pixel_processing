import numpy as np
import math
import pandas as pd
import json

class Camera_Calibration(object):
    def __init__(self, df_points, df_worlds, nb_image):
        # Initiating my global variables
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
        # Get the number of columns of images points you want to use
        images_titles = self.df.columns.tolist()[:self.nb_image*2]
        images_names = [f"img{i}" for i in range(1, self.nb_image+1)]
        images_list = []
        # Generating a list of data points to create a dictionnary to store the images points (key -> image name, value -> points positions)
        for item in images_titles:
            images_list.append(self.df[item])
        count = 0
        for i in range(0, len(images_list)-1, 2):
            # Zip will combine my x axis points with my y axis points to form a (x, y) point coordinates
            result = zip(images_list[i], images_list[i+1])
            self.images_points[images_names[count]] = list(result)
            count += 1
        # This will initiate the world points coordinates as a list of tuples
        self.world_points = list(zip(self.df_worlds["worldPoints_x"], self.df_worlds["worldPoints_y"]))
        pass

    def h_matrix(self):
        images_names = [f"img{i}" for i in range(1, self.nb_image+1)]
        count = 0
        # Iterating over each array of pixels positions of the images to constrcut my H arrays
        for _, points in self.images_points.items():
            # Storing the transposed parameters to obtain my H matrix
            linear_in_params = []
            for i in range(len(points)): 
                # Setting up the linear parameters for each points of the images
                xi, yi, Xi, Yi = points[i][0], points[i][1], self.world_points[i][0], self.world_points[i][1]
                first_row = [0, 0, 0, Xi, Yi, 1, -yi*Xi, -yi*Yi, -yi]
                second_row = [Xi, Yi, 1, 0, 0, 0, -xi*Xi, -xi*Yi, -xi]
                linear_in_params.append(first_row)
                linear_in_params.append(second_row)
            self.stored_phi_mat[images_names[count]] = linear_in_params
            count += 1
            
        count = 0
        for _, value in self.stored_phi_mat.items():
            self.stored_h_mat[images_names[count]] = self.get_null_space(value)
            count += 1
        pass
    
    def get_null_space(self, matrix):
        _, S, V = np.linalg.svd(matrix)
        # Finding the index of the lowest value of S and getting the value at this index in V
        smallest_value = math.inf
        index_s_value = 0
        for i in range(len(S)):
            if S[i] < smallest_value:
                smallest_value = S[i]
                index_s_value = i
        vector = V[index_s_value]
        return vector

    def b_matrix(self):
        all_images_v_vectors = []
        # Storing my v12, v11 and v22 vectors
        for _, value in self.stored_h_mat.items():
            h = np.array(value).reshape(3,3)
            vectors = { "v12": None, "v11": None, "v22": None }
            index_vectors = [[0,1], [0,0], [1,1]]
            counter = 0
            for key in vectors:
                i, j = index_vectors[counter][0], index_vectors[counter][1]
                vectors[key] = np.array([ h[0][i]*h[0][j], h[1][i]*h[0][j] + h[0][i]*h[1][j],
                                          h[1][i]*h[1][j], h[2][i]*h[0][j] + h[0][i]*h[2][j], 
                                          h[2][i]*h[1][j] + h[1][i]*h[2][j], h[2][i]*h[2][j] ])
                counter += 1
            all_images_v_vectors.append(vectors["v12"])
            all_images_v_vectors.append(np.subtract(vectors["v11"], vectors["v22"]))
        b_vector = self.get_null_space(all_images_v_vectors)
        return b_vector
    
    def intrinsic_params(self, b_vector):
        # Setting all my parameters for the calculations of the intrinsics params
        b11, b12, b22, b13, b23, b33 = b_vector
        y = (b12*b13-b11*b23)/(b11*b22-b12**2)
        lambda_val = b33 - (b13**2+y*(b12*b13-b11*b23))/b11
        alpha_val = math.sqrt((lambda_val/b11))
        beta_val = math.sqrt(abs(lambda_val*b11)/abs(b11*b22-b12**2))
        gamma_val = -(b12*(alpha_val**2)*beta_val)/lambda_val
        x = (gamma_val*y)/alpha_val - (b13*(alpha_val**2))/lambda_val
        np.set_printoptions(suppress=True)
        self.K = np.array(np.around(np.array([alpha_val, gamma_val, x, 0, beta_val, y, 0, 0, 1]).astype(np.float64), decimals=5)).reshape(3,3)
        return self.K

    def extrinsics_params(self):
        for key, value in self.stored_h_mat.items():
            reshape_value = np.array(value).reshape(3,3)
            K_inverse = np.linalg.inv(self.K)
            scale = 1/np.linalg.norm(np.matmul(K_inverse, reshape_value[:,0]))
            r1 = np.matmul(scale * K_inverse, reshape_value[:,0])
            r2 = np.matmul(scale * K_inverse, reshape_value[:,1])
            r3 = np.cross(r1, r2)
            P_org = np.matmul(scale * K_inverse, reshape_value[:,2])
            np.set_printoptions(suppress=True)
            extrinsincs_par = np.ones(shape=(3, 4))
            for i in range(3):
                extrinsincs_par[i][0] = r1[i]
                extrinsincs_par[i][1] = r2[i]
                extrinsincs_par[i][2] = r3[i]
                extrinsincs_par[i][3] = P_org[i]
            self.extrinsics_parameters[key] = np.around(extrinsincs_par.astype(np.float64), decimals=5)
        return self.extrinsics_parameters

#### Lines executed to obtain from the object image the corresponding results for the Tcw and K matrices ####

# Fetching the images data points from the matlab file stored in a csv file
data = pd.read_csv("./datasets/image_points_1.csv", sep=";", encoding="utf_8")
data_world = pd.read_csv("./datasets/world_points_1.csv", sep=";", encoding="utf_8")
df = pd.DataFrame(data)
df_world = pd.DataFrame(data_world)

nb_image = 6
camera = Camera_Calibration(df, df_world, nb_image)
camera.h_matrix()
b_vector = camera.b_matrix()
K = camera.intrinsic_params(b_vector)
E_params = camera.extrinsics_params()

# Results being print out with those comamnds
print("These are the intrinsic parameters of the camera:", "\n", K, "\n")
print("These are the extrinsics parameters of the first image:", "\n", E_params["img1"])

# Save as a JSON object the K and Tcw matrices
save_K = json.dumps({ "K_matrix": list(K.flatten())})
save_Tcw = json.dumps({ "Tcw_matrix": list(E_params["img1"].flatten())})
with open("./K.json", "w") as K_file:
    K_file.write(f"{save_K}")
K_file.close()
with open("./Tcw.json", "w") as Tcw_file:
    Tcw_file.write(f"{save_Tcw}")
Tcw_file.close()
