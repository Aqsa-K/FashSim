from skimage import data, io, filters
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.util import crop
import os
import numpy as np

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = np.array(images, dtype='float32')
    return images_hr

def normalize(input_data):
    x = input_data.astype(np.float32)
    x_norm = (x-127.5)/127.5
    return x_norm

def load_training_data(directory, hr_image_shape):

    #read images into x_train 
    x_train = []
    file_name_list = []

    for file_name in os.listdir(directory):
        img_path = os.path.join(directory, file_name)
        img = data.imread(img_path)
        img = resize(img, hr_image_shape , anti_aliasing=True)
        x_train.append(img)
        file_name_list.append(file_name)

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)

    return x_train_hr, file_name_list

def crop_image(img, h1=0.1, w1=0.2):
    delta_h = img.shape[0]*h1
    delta_w = img.shape[1]*w1
    
    print(delta_h, delta_w)
    
    cropped_img = crop(img, ((delta_h*2, delta_h), (delta_w, delta_w), (0,0)), copy=False)
    
    return cropped_img


#Image Patches


from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d

save_directory = 'data/clothes_patches/'



def create_image_patches(directory, save_directory, file_name_list):
    
    for file_name in file_name_list:
        img_path = os.path.join(directory, file_name)
        img = data.imread(img_path)
        cropped_img = crop_image(img)
        try:
            img_patches = extract_patches_2d(cropped_img, (224,224), max_patches=10)
        except Exception as e:
            cropped_img = crop_image(img, 0.1,0.1)
            img_patches = extract_patches_2d(cropped_img, (224,224), max_patches=10)
            
        io.imshow(cropped_img)
        plt.show()
        save_image_patches(save_directory, file_name, img_patches)
        
        k=11
        for img_patch in img_patches:
            smaller_img_patches = extract_patches_2d(cropped_img, (100,100), max_patches=2)
            save_image_patches(save_directory, file_name, smaller_img_patches, k)
            k+=2


        for patch in img_patches:
            io.imshow(patch)
            plt.show()



def save_image_patches(directory, file_name, image_patches, k=1):
    
    print("SAVING: ", file_name)
    try:
        for patch in image_patches:
            x = file_name.split('.')
            x = x[len(x)-2]
            patch_file_name = x+ '_{}.jpg'.format(str(k))
            img_path = os.path.join(directory, patch_file_name)
            io.imsave(img_path, patch)
            k+=1
    except Exception as e:
        print("Could not save for {}".format(file_name))



def get_image_patches(img, max_patches=10):

    img_patches = extract_patches_2d(img, (224,224), max_patches=10)
    io.imshow(img)
    plt.show()

    for patch in img_patches:
        io.imshow(patch)
        plt.show()
        
    return img_patches
    
def get_img_embedding(img):
    
    y = vgg_model.model.predict(img)
    
    return y


import re

def get_orig_patch_file(patch_filename):
    orig_file_name = re.sub("\_\d+\.jpg$", '', patch_filename)
    orig_file_name = orig_file_name + ('.jpg')
    
    return orig_file_name
    
def replace_file_format(file_name, frmt = '.png'):
    return file_name.replace('.jpg', frmt)


import json

def save_json(version, emb_dict):
    with open("embeddings_{}".format(version), 'w') as fp:
        json.dump(emb_dict, fp)


def save_embeddings(embeddings, file_name_list):
    embedding_dict = {}
    for file_name, embedding in zip(file_name_list, embeddings):
        embedding_dict[file_name] = embedding.tolist()
#         b = a.tolist() # nested lists with same data, indices
# file_path = "/path.json" ## your path variable
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    
    return embedding_dict



def display_images(file_name_list, directory, end_idx, start_idx=0):
    for file_name in file_name_list[start_idx:end_idx]:
        img_path = os.path.join(directory, file_name)
        img = data.imread(img_path)
        io.imshow(img)
        plt.show()