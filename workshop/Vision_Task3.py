from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import numpy as np
import json

class P_Processing(object):
    def __init__(self, img):
        self.img = img
        self.edges_detection_pixels = []
        pass

    def reshape_img(self):
        # Converts the images into a 2D array of grey pixels intensity
        I = np.array((self.img.convert("L").getdata()))
        I_reshape = []
        width, height = self.img.size
        count = 0
        # Looping here will allow to create a 3D array of pixels intensity of dimensions (640, 480, 3)
        for _ in range(height):
            inner_list = []
            for _ in range(width):
                inner_list.append(I[count])
                count += 1
            I_reshape.append(inner_list)
        return np.array(I_reshape)

    def transform_g_to_rgb(self, matrix):
        # Obtaining RGB matrices with the grey value copy three times to keep the grey-scale color
        rgb_pixels = []
        for i in range(len(matrix)):
            inner_list = []
            for j in range(len(matrix[0])):
                item = [matrix[i][j],
                        matrix[i][j],
                        matrix[i][j]]
                inner_list.append(item)
            rgb_pixels.append(inner_list)
        return rgb_pixels

    def find_corners(self):
        # Creating a threshold that will be updated after analysing the pixel intensity
        dict_intensity = {}
        # Finding the biggest value for the edge detections (corners)
        for i in range(self.edges_detection_pixels.shape[0]):
            for j in range(self.edges_detection_pixels.shape[1]):
                if int(self.edges_detection_pixels[i][j]) in dict_intensity:
                    dict_intensity[int(self.edges_detection_pixels[i][j])] += 1
                else:
                    dict_intensity[int(self.edges_detection_pixels[i][j])] = 1
        x_val = list(dict_intensity.keys())
        y_val = list(dict_intensity.values())
        plt.bar(x_val, y_val, width=0.8, align="center")
        plt.xlim([0, 255])
        plt.title("Proportion of intensity pixel for corners detection")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Number of apparition")
        plt.show()
        # Lets keep count of the number of corners
        corners_count = 0
        # Turning my Grey scale image back to an RGB image to be able to color in red the edges
        low_threshold, high_threshold = 145, 148
        rgb_array = self.transform_g_to_rgb(self.edges_detection_pixels)
        for i in range(len(rgb_array)):
            for j in range(len(rgb_array[0])):
                if rgb_array[i][j][0] > low_threshold and rgb_array[i][j][0] < high_threshold:
                    # Turning red the pixels higher than the threshold
                    rgb_array[i][j][0] = 255
                    rgb_array[i][j][1] = 0
                    rgb_array[i][j][2] = 0
                    # Increment corners
                    corners_count += 1
        # De comment this section
        for i in range(len(rgb_array)):
            for j in range(len(rgb_array[0])):
                if rgb_array[i][j][0] != 255 and sum(rgb_array[i][j])/3 > 100:
                    # Putting in darker value the edges to see the corners points
                    rgb_array[i][j][0] = 40
                    rgb_array[i][j][1] = 40
                    rgb_array[i][j][2] = 40
        self.img = Image.fromarray(np.uint8(rgb_array))
        return corners_count

    def get_edges(self):
        # Obtaining an array of dimension (640, 480, 3)
        I_grey = self.reshape_img()
        # Performing the sobel horizontal and vertical edge detection
        prewit_y_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewit_x_filter = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewit_x_mat = np.sqrt(np.square(self.get_sobel(prewit_x_filter, I_grey)))
        prewit_y_mat = np.sqrt(np.square(self.get_sobel(prewit_y_filter, I_grey)))
        combined_mat = np.sqrt(np.add(np.square(prewit_x_mat), np.square(prewit_y_mat)))
        # Taking the median to remove residual noise
        self.edges_detection_pixels = self.median_value(combined_mat)
        # Recreating the image from the above array
        self.img = Image.fromarray(np.uint8(self.edges_detection_pixels))
        pass

    def get_sobel(self, mask, I):
        # Applying the sobel filter and creating a new intensity matrix of pixels
        intensity = []
        for i in range(1, len(I)-1):
            inner_list = []
            for j in range(1, len(I[0])-1):
                new_value = I[i-1][j-1]*mask[0][0] + I[i-1][j]*mask[0][1] + I[i-1][j+1]*mask[0][2] + I[i][j-1]*mask[1][0] \
                        + I[i][j]*mask[1][1] + I[i][j+1]*mask[1][2] + I[i+1][j-1]*mask[2][0] + I[i+1][j]*mask[2][1] + I[i+1][j+1]*mask[2][2]
                inner_list.append(abs(new_value))
            intensity.append(inner_list)
        return np.array(intensity)

    def median_value(self, I):
        intensity = []
        for i in range(1, len(I)-1):
            inner_list = []
            for j in range(1, len(I)-1):
                new_value = np.median([I[i-1][j-1], I[i-1][j], I[i-1][j+1], I[i][j-1], \
                    I[i][j], I[i][j+1], I[i+1][j-1], I[i+1][j], I[i+1][j+1]])
                inner_list.append(new_value)
            intensity.append(inner_list)
        return np.array(intensity)

    def get_center_orientation(self):
        # Obtaining an array of dimension (640, 480, 3)
        I = self.reshape_img()
        # Transforming my RGB to a grey-scale
        threshold = 80
        I_threshold = [[0 if x > threshold else 255 for x in item] for item in I]
        # Calculating my needed variables for finding the center position
        M00 = sum(np.array(I_threshold, dtype=np.float64).flatten())
        M01 = sum(np.array([[I_threshold[i][j]*j for j in range(len(I[0]))] for i in range(len(I))], dtype=np.float64).flatten())
        M10 = sum(np.array([[I_threshold[i][j]*i for j in range(len(I[0]))] for i in range(len(I))], dtype=np.float64).flatten())
        Xc = M10/M00
        Yc = M01/M00
        # Finding the eigen values and eigen vectors
        eigenval, eigenvect = self.get_inertia_eig(I_threshold, Xc, Yc)
        a = 2*math.sqrt(max(eigenval)/M00)
        b = 2*math.sqrt(min(eigenval)/M00)
        vect1, vect2 = eigenvect
        # Calculating the cosine between my two vectors and doing the arccos of it to find the tetha angle
        dot_product = np.dot(vect1, vect2)
        length_a = math.sqrt(vect1[0]**2 + vect1[1]**2)
        length_b = math.sqrt(vect2[0]**2 + vect2[1]**2)
        cos_tetha = dot_product/(length_a*length_b)
        tetha = np.arccos(cos_tetha)

        # Calculating the cosine between the vector of radius "a" and the horizontal vector
        horizontal_vect = (-1,0)
        dot_product = np.dot(vect1, horizontal_vect)
        length_a = math.sqrt(vect1[0]**2 + vect1[1]**2)
        length_b = math.sqrt(horizontal_vect[0]**2 + horizontal_vect[1]**2)
        cos_tetha = dot_product/(length_a*length_b)
        tetha_horizontal = np.arccos(cos_tetha)
        center = (abs(Xc), abs(Yc))
        # Initiating an rgb array to color the axis drawn (grey values will just fill up the rgb values, i.e. [128, 128, 128])
        rgb_array = self.transform_g_to_rgb(I_threshold)
        self.img = Image.fromarray(np.uint8(rgb_array))
        # Calling the library ImageDraw to be able to draw a line between two points without having to color every pixels
        draw = ImageDraw.Draw(self.img)
        # Plotting the line for the b minor axis radius
        xb1 = center[0] - b * math.cos(math.degrees(tetha))
        yb1 = center[1] - b * math.sin(math.degrees(tetha))
        xb2 = center[0] 
        yb2 = center[1] 
        draw.line((xb1, yb1, xb2, yb2), fill=(0, 230, 0), width=2)
        # Plotting the line for the a major axis radius
        xa1 = center[0] + a * math.cos(math.degrees(tetha))
        ya1 = center[1] + a * math.cos(math.degrees(tetha))
        xa2 = center[0] 
        ya2 = center[1] 
        draw.line((xa1, ya1, xa2, ya2), fill=(230, 0, 0), width=2)
        # Drawing the center with multiple points to see it clearly from our center coordinates
        number_points = 6
        for i in range(number_points):
            for j in range(number_points):
                draw.point((center[0]+i, center[1]+j), fill=(0, 0, 255))
        return center, round(a,2), round(b,2), -math.degrees(tetha_horizontal), math.degrees(tetha)

    def get_inertia_eig(self, I, X, Y):
        J = np.empty(shape=(2, 2))
        # These values in my matrix will store the constants for calculating each μ of my J matrix
        values = [[(2, 0), (1, 1)], [(1, 1), (0, 2)]]
        for k in range(len(values)):
            for l in range(len(values[0])):
                μ = np.sum(np.array([[I[i][j] * pow((i-X), values[k][l][0]) * pow((j-Y), values[k][l][1]) for j in range(len(I[0]))] for i in range(len(I))]))
                J[k][l] = μ
        eigenval, eigenvect = np.linalg.eig(J)
        return eigenval, eigenvect

    def get_circularity(self):
        # Obtaining an array of dimension (640, 480, 3)
        I_grey = self.reshape_img()
        # Transforming my RGB to a grey-scale
        threshold = 120
        I_threshold = [[0 if x > threshold else 255 for x in item] for item in I_grey]
        # Looping through the pixels to find the perimeter of the shape
        # We'll look at all the variations of intensity as for the edges but this will detect a very fine line of pixels drawing the perimeter of the object
        pos = []
        # We'll use the grey scale intensity matrix to compute the area (M00)
        area = sum(np.array(I_grey).flatten())
        perimeter_pixels = 0
        for i in range(len(I_threshold)):
            for j in range(len(I_threshold[0])-1):
                if (I_threshold[i][j] == 0 and I_threshold[i][j+1] == 255) or (I_threshold[i][j] == 255 and I_threshold[i][j+1] == 0):
                    perimeter_pixels += I_threshold[i][j]
                    pos.append((i,j))
        for i in range(len(I_threshold)):
            for j in range(len(I_threshold[0])):
                I_threshold[i][j] = 0
        for item in pos:
            I_threshold[item[0]][item[1]] = 255
        circularity = (4 * math.pi * area) / (perimeter_pixels ** 2)
        self.img = Image.fromarray(np.uint8(I_threshold))
        return circularity

image = "./assign_files/Object.png"
img = Image.open(image)

#### Lines to execute to obtain three different images with calculated values for all the different characteristics of the object ####

edge_corner_img = P_Processing(img)
edge_corner_img.get_edges()
edge_corner_img.img.show()
nb_edges = edge_corner_img.find_corners()
image_corner_edges = edge_corner_img.img
image_corner_edges.show()
# image_corner_edges.save("./vision_task3_results/edges.png")
# image_corner_edges.save("./vision_task3_results/corners.png")
print("The number of corners is:", nb_edges)

center_vectors_img = P_Processing(img)
center, a_radius, b_radius, tetha_h, tetha = center_vectors_img.get_center_orientation()
image_center_orientation = center_vectors_img.img
image_center_orientation.show()
# image_center_orientation.save("./vision_task3_results/center_orientation.png")
print("The center of the object is:", f"({round(center[0],2)}, {round(center[1],2)})")
print("The major and minor radius of the object are:", a_radius, "and", b_radius)
print("The angle of orientation is:", f"{round(tetha_h,2)}°")
print("The angle between the two vectors is:", f"{tetha}°")

circularity_img = P_Processing(img)
circularity_value = circularity_img.get_circularity()
if circularity_value >= 0.85 and circularity_value <= 1.0:
    print("This is a circle-like shape with a roundness of:", circularity_value)
elif circularity_value >= 0.7 and circularity_value <= math.pi/4:
    print("This is a square-like shape with a roundness of:", circularity_value)
else:
    print("This is an elongated-like shape with a roundness of:", circularity_value)
image_circularity = circularity_img.img
image_circularity.show()
# image_circularity.save("./vision_task3_results/perimeter.png")

save_PObject = json.dumps({ "PObject_matrix": [round(center[0],2), round(center[1],2), 0]})
with open("./PObject.json", "w") as Tcw_file:
    Tcw_file.write(f"{save_PObject}")
Tcw_file.close()