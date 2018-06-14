"""
    checkdata.py
    draw image from data
"""
import numpy as np
from PIL import Image

"""
    fig: image size
    output_dir: output image path
    common_path: common path of input data
"""
fig = 28
output_dir = "images"
common_path = "../mnist"

data = np.load(common_path+"/mnist_train/data_15000.npy")
label = np.load(common_path+"/mnist_train/label_15000.npy")

for i in range(len(data)):
    img = Image.fromarray(data[i].reshape(fig, fig))
    img.save(output_dir+'/'+str(i)+"_"+str(label[i])+".jpg")
