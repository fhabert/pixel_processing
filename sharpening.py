from PIL import Image
import numpy as np

def sharpening(self):
    loc = self.img
    img = Image.open(loc)
    img_resized = img.resize((self.pixels, self.pixels))
    I_ = self.resize_img(img_resized, self.pixels)
    I = [[sum(x)/3 for x in j] for j in I_]
    intensity = []
    mask = np.array([[-1/9, -1/9, -1/9], [-1/9, 17/9, -1/9], [-1/9, -1/9, -1/9]])         
    for i in range(1, len(I)-1):
        inner_list = []
        for j in range(1, len(I[0])-1):
            new_value = I[i-1][j-1]*mask[0][0] + I[i-1][j]*mask[0][1] + I[i-1][j+1]*mask[0][2] + I[i][j-1]*mask[1][0] \
                    + I[i][j]*mask[1][1] + I[i][j+1]*mask[1][2] + I[i+1][j-1]*mask[2][0] + I[i+1][j]*mask[2][1] + I[i+1][j+1]*mask[2][2]
            inner_list.append(abs(new_value))
        intensity.append(inner_list)
    new_img = Image.fromarray(np.uint8(intensity))
    new_img.show()
    return intensity

sharpening()