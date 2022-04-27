import numpy as np
from PIL import Image

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img.resize((256,256),Image.BICUBIC)
    return img

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def save_img(img_tensor,filename):
    img_numpy = img_tensor.float().numpy()
    img_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    img_numpy = img_numpy.clip(0,255).astype(np.uint8)
    img = Image.fromarray(img_numpy)
    img.save(filename)
    print("Image saved as {}".format(filename))






