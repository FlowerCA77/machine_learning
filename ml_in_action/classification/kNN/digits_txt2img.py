from PIL import Image, ImageFilter
import time
import numpy as np

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    pixel_array = np.array([[int(char) for char in line] for line in lines], dtype=np.uint8)
    return pixel_array

def create_image_from_array(pixel_array):
    img = Image.fromarray((1 - pixel_array) * 255)
    return img

def create_smooth_image(img, size=(128, 128), blur_radius=(1.4, 6.2)):
    img_large = img.resize(size, resample=Image.NEAREST)
    
    np.random.seed(int(1000*np.float64(time.time()))%(42**4))
    blur_radius = np.random.randint(blur_radius[0], blur_radius[1])
    img_smooth = img_large.filter(ImageFilter.GaussianBlur(blur_radius))
    
    return img_smooth

def save_image(img, output_path):
    img.save(output_path)

def main(txt_file, output_image_path):
    pixel_array = read_txt_file(txt_file)
    img_32x32 = create_image_from_array(pixel_array)
    
    img_128x128 = create_smooth_image(img_32x32)
    
    save_image(img_128x128, output_image_path)

K = np.random.randint(1,10,(8,1))

for i in range(10):
    for k in K:
        input_txt = '/home/flower/ai_projects/machine_learning/ml_in_action/classification/kNN/testDigits/' + str(i) + '_' + str(k[0]) + '.txt'
        output_image = '/home/flower/ai_projects/machine_learning/ml_in_action/classification/kNN/imageDigits/' + str(i) + '_' + str(k[0]) + '.png'
        main(input_txt, output_image)
