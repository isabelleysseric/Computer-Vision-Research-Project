# PYTHON 3.8

"""
    INFERENCE DEPTH ALL
"""


# To install yacs, gdown, matplotlib, opencv-python

import os, time
import importlib
import imageio.v3 as iio
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2



##################################
##      Download pretrained     ##
##################################

"""
- We use HoHoNet w/ hardnet encoder in this demo
- Download other version [here](https://drive.google.com/drive/folders/1raT3vRXnQXRAQuYq36dE-93xFc_hgkTQ?usp=sharing)
"""

PRETRAINED_PTH = './HoHoNet/ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth'

if not os.path.exists(PRETRAINED_PTH):
    os.makedirs(os.path.split(PRETRAINED_PTH)[0], exist_ok=True)
    !gdown 'https://drive.google.com/uc?id=1w31WOzQvvGjoCXXFmnkeOL035hx7NJRQ' -O $PRETRAINED_PTH



##################################
##       Download image         ##
##################################

"""
- We use a out-of-distribution image from PanoContext
"""

if not os.path.exists('./HoHoNet/assets/pano_asmasuxybohhcj.png'):
    !gdown 'https://drive.google.com/uc?id=1CXl6RPK6yPRFXxsa5OisHV9KwyRcejHu' -O 'assets/pano_asmasuxybohhcj.png'

# Create dataset
path_dir = './HoHoNet/assets/Pano/'
dataset = sorted(glob.glob(os.path.join(path_dir, '*jpg')))

color_map = np.array([
        [  0,   0,   0],
        [255,   0,  40],
        [255,  72,   0],
        [255, 185,   0],
        [205, 255,   0],
        [ 91, 255,   0],
        [  0, 255,  21],
        [  0, 255, 139],
        [  0, 255, 252],
        [  0, 143, 255],
        [  0,  23, 255],
        [ 90,   0, 255],
        [204,   0, 255],
        [255,   0, 191]], dtype=np.uint8)

# Visualize an example
bgr = cv2.imread(os.path.join('./HoHoNet/panocontext/pano_asmasuxybohhcj.png'))
rgb = bgr[:,:,::-1]
plt.imshow(rgb)
plt.show()



##################################
##      Load model config       ##
##################################

"""
- We use HoHoNet w/ hardnet encoder in this demo
- Find out other version in `mp3d_depth/` and `s2d3d_depth`
"""

# Load model config
from lib.config import config

config.defrost()
config.merge_from_file('./HoHoNet/config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml')
config.model.kwargs['backbone_config']['kwargs']['pretrained'] = False
config.freeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

# Load model
model_file = importlib.import_module(config.model.file)
model_class = getattr(model_file, config.model.modelclass)
net = model_class(**config.model.kwargs)
net.load_state_dict(torch.load(PRETRAINED_PTH, map_location=device))
net = net.eval().to(device)

# Visualize result in 2d
for i in dataset:

    # Extract name
    name = i.replace('.jpg', '').replace('./HoHoNet/assets/Pano', '')
    name1 = name.replace('\\', '')
    name2 = name1.replace(' ', '_')
    print('name1: ', name1)

    # Resize panoramic
    bgr = cv2.imread(os.path.join(i))
    rgb = bgr[:,:,::-1]
    rgb_resize = cv2.resize(rgb, (1024, 512), interpolation = cv2.INTER_AREA)

    # Move image into tensor, normalize to [0, 255], resize to training resolution
    x = torch.from_numpy(rgb_resize).permute(2,0,1)[None].float() / 255.
    if x.shape[2:] != config.dataset.common_kwargs.hw:
        x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode='area')
    x = x.to(device)

    # Model feedforward
    with torch.no_grad():
        ts = time.time()
        pred_depth = net.infer(x)
        if not torch.is_tensor(pred_depth):
            pred_depth = pred_depth.pop('depth')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f'Eps time: {time.time() - ts:.2f} sec.')


    # Visualize result in 2d
    image1 = rgb[80:-80]
    image2 = pred_depth.squeeze()[80:-80].cpu().numpy()

    plt.figure(figsize=(15,6))

    plt.subplot(121)
    plt.imshow(image1)
    plt.axis('off')
    plt.title('RGB')

    plt.subplot(122)
    plt.imshow(image2, cmap='inferno_r')#, vmin=0.1, vmax=10)
    plt.axis('off')
    plt.title('Depth prediction')

    # Save depth prediction
    # plt.imsave(f'./HoHoNet/assets/output/depth/{name}_depth.png', image2, format='png', cmap='inferno_r', vmin=0.1, vmax=10)

    # Save figure
    # plt.draw()
    # plt.savefig(f'./HoHoNet/assets/output/result/depth/{name}_result_depth.png', format='png')
    plt.show()