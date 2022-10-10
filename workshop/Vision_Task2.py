from operator import length_hint
from PIL import Image, ImageDraw
import math
import numpy as np

class P_Processing(object):
    def __init__(self, img):
        self.img = img
        self.edges_detection_pixels = []
        pass

    def reshape_img(self):
        # Converts the images into a 2D array of pixels intensity (number of pixels, 3)
        I = np.array((self.img.convert("RGB").getdata()))
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

    def transform_g_to_rgb(self, array):
        # Obtaining RGB matrices with the grey value copy three times to keep the grey-scale color
        rgb_pixels = []
        for i in range(len(array)):
            inner_list = []
            for j in range(len(array[0])):
                item = [array[i][j],
                        array[i][j],
                        array[i][j]]
                inner_list.append(item)
            rgb_pixels.append(inner_list)
        return rgb_pixels

    def find_corners(self):
        pos = []
        max_value = 0
        # Finding the biggest value for the edge detections (corners)
        for i in range(self.edges_detection_pixels.shape[0]):
            for j in range(self.edges_detection_pixels.shape[1]):
                if self.edges_detection_pixels[i][j] > max_value:
                    max_value = self.edges_detection_pixels[i][j]
                    pos.append(max_value)
        # Creating a threshold in the last third of my biggest values
        corner_threshold = int(sorted(pos)[int(len(pos)*2/3)])
        # Turning my Grey scale image back to an RGB image to be able to color in red the edges
        rgb_array = self.transform_g_to_rgb(self.edges_detection_pixels)
        for i in range(len(rgb_array)):
            for j in range(len(rgb_array[0])):
                if rgb_array[i][j][0] > corner_threshold:
                    # Turning red the pixels higher than the threshold
                    rgb_array[i][j][0] = 255
                    rgb_array[i][j][1] = 0
                    rgb_array[i][j][2] = 0
        self.img = Image.fromarray(np.uint8(rgb_array))
        pass

    def get_edges(self):
        # Obtaining an array of dimension (640, 480, 3)
        I = self.reshape_img()
        # Transforming my RGB to a black and white threshold pixels
        I_grey = [[sum(x)/3 for x in j] for j in I]
        # Performing the sobel horizontal and vertical edge detection
        sobel_y_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_x_filter = sobel_y_filter.T
        sobel_x_mat = np.sqrt(np.square(self.get_sobel(sobel_x_filter, I_grey)))
        sobel_y_mat = np.sqrt(np.square(self.get_sobel(sobel_y_filter, I_grey)))
        self.edges_detection_pixels = np.sqrt(np.add(np.square(sobel_x_mat), np.square(sobel_y_mat)))
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

    def get_center_orientation(self):
        # Obtaining an array of dimension (640, 480, 3)
        I = self.reshape_img()
        # Transforming my RGB to a grey-scale
        threshold = 80
        I_threshold = [[0 if sum(x)/3 > threshold else 255 for x in item] for item in I]
        # Calculating my needed variables for finding the center position
        M00 = sum(np.array(I_threshold).flatten())
        M01 = sum(np.array([[I_threshold[i][j]*j for j in range(len(I_threshold[0]))] for i in range(len(I_threshold))], dtype=np.float64).flatten())
        M10 = sum(np.array([[I_threshold[i][j]*i for j in range(len(I_threshold[0]))] for i in range(len(I_threshold))], dtype=np.float64).flatten())
        Xc = M10/M00
        Yc = M01/M00
        # Finding the eigen values and eigen vectors
        eigenval, eigenvect = self.get_inertia_eig(I_threshold, Xc, Yc)
        a = 2*math.sqrt(max(eigenval)/M00)
        b = 2*math.sqrt(min(eigenval)/M00)
        vect1, vect2 = eigenvect
        # Calculating the cosine between my two vectors and doing the arcos of it to find the tetha angle
        dot_product = np.dot(vect1, vect2)
        length_a = math.sqrt(vect1[0]**2 + vect1[1]**2)
        length_b = math.sqrt(vect2[0]**2 + vect2[1]**2)
        cos_tetha = dot_product/length_a*length_b
        tetha = np.arccos(cos_tetha)
        center = (abs(Xc), abs(Yc))
        rgb_array = self.transform_g_to_rgb(I_threshold)
        self.img = Image.fromarray(np.uint8(rgb_array))
        draw = ImageDraw.Draw(self.img)
        # Plotting the line for the b minor axis radius
        xb1 = center[0] + b * math.cos(math.degrees(tetha))
        yb1 = center[1] + b * math.sin(math.degrees(tetha))
        xb2 = center[0] - b * math.cos(math.degrees(tetha))
        yb2 = center[1] - b * math.sin(math.degrees(tetha))
        draw.line((xb1, yb1, xb2, yb2), fill=(0, 230, 0), width=2)
        # Plotting the line for the a major axis radius
        xa1 = center[0] + a * math.cos(math.degrees(tetha)+90)
        ya1 = center[1] + a * math.sin(math.degrees(tetha)+90)
        xa2 = center[0] - a * math.cos(math.degrees(tetha)+90)
        ya2 = center[1] - a * math.sin(math.degrees(tetha)+90)
        draw.line((xa1, ya1, xa2, ya2), fill=(230, 0, 0), width=2)
        # Drawing the center with multiple points to see it clearly from our center coordinates
        number_points = 6
        for i in range(number_points):
            for j in range(number_points):
                draw.point((center[0]+i, center[1]+j), fill=(0, 0, 255))
        print("These are the coordinates of my center:", center)
        pass

    def get_inertia_eig(self, I, X, Y):
        J = np.empty(shape=(2, 2))
        # These values in my matrix will store the constants for calculating each μ of my J matrix
        values = [[(2, 0), (1, 1)], [(1, 1), (0, 2)]]
        for k in range(len(values)):
            for l in range(len(values[0])):
                μ = sum(np.array([[I[i][j]*((i-X)**values[k][l][0])*((j-Y)**values[k][l][1]) for j in range(len(I[0]))] for i in range(len(I))]).flatten())
                J[k][l] = μ
        eigenval, eigenvect = np.linalg.eig(J)
        return eigenval, eigenvect

    def get_circularity(self):
        # Obtaining an array of dimension (640, 480, 3)
        I = self.reshape_img()
        # Transforming my RGB to a grey-scale
        threshold = 120
        I_threshold = [[0 if sum(x)/3 > threshold else 255 for x in item] for item in I]
        # Looping through the pixels to find the perimeter of the shape
        # We'll look at all the variations of intensity as for the edges but this will detect a very fine line of pixels drawing the perimeter of the object
        pos = []
        # We'll use the grey scale intensity matrix to compute the are (M00)
        I_grey = [[sum(x)/3 for x in j] for j in I]
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

# edge_corner_img = P_Processing(img)
# edge_corner_img.get_edges()
# edge_corner_img.find_corners()
# image_corner_edges = edge_corner_img.img
# image_corner_edges.show()
# image_corner_edges.save("./resulted_images/corner_edges.png")

# center_vectors_img = P_Processing(img)
# center_vectors_img.get_center_orientation()
# image_center_orientation = center_vectors_img.img
# image_center_orientation.show()
# image_center_orientation.save("./resulted_images/center_orientation.png")

# circularity_img = P_Processing(img)
# circularity_value = circularity_img.get_circularity()
# if circularity_value >= 0.85 and circularity_value <= 1.0:
#     print("This is a circle-like shape with a roundness of:", circularity_value)
# elif circularity_value >= 0.7 and circularity_value <= math.pi/4:
#     print("This is a square-like shape with a roundness of:", circularity_value)
# else:
#     print("This is an elongated-like shape with a roundness of:", circularity_value)
# image_circularity = circularity_img.img
# image_circularity.show()
# image_circularity.save("./resulted_images/perimeter.png")
