import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from VGG19 import VGG_MODEL
from utils import *
from similarity import *


np.random.seed(10)
downscale_factor = 4
image_shape = (224,224,3)
train_directory = 'data/Clothes_data/'
patch_directory = 'data/clothes_patches/'
split_idx = 500

# Make an instance of the VGG class
vgg_model = VGG_MODEL(image_shape) 

vgg_model.model.summary()

# Get complete images and their names lists

x_images, file_name_list = load_training_data(train_directory, image_shape)
y = vgg_model.model.predict(x_images)

y_train = y[:split_idx]
y_test = y[split_idx:]

print("Finished")

from sklearn.neighbors import NearestNeighbors
img_nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(y_train)
# distances, indices = img_nbrs.kneighbors(y_train)

# Get top similar clothes based on full length images and knn algorithm
get_similar_top_n(img_nbrs, y_test, file_name_list[split_idx:], file_name_list, train_directory, 2)



# we can also use patches to retrieve similar clothes
# patches are first created for all the images and stored in a folder
# patches can be created using the following function

# create_image_patches(train_directory, save_directory, file_name_list)
# This function will create image patches for all the images in the train_directory
# and save the patches in the save_directory
# Note: because we have done this step already, our pacthes are present in the patch_directory
# We cwill now see how to retrieve images based on similar patches

#----------------PATCHES--------------------
# Get image patches and their file names list
patch_images, patch_file_list = load_training_data(patch_directory, image_shape)
y_patches = vgg_model.model.predict(patch_images)

patches_split = 1200
y_patches_train = y_patches[:patches_split]
y_patches_test = y_patches[patches_split:]

# save embeddings
# embedding_dict = save_embeddings(y_patches, patch_file_list)
# embedding_dict.keys()
# save_json('patches_1', embedding_dict)

display_images(patch_file_list, patch_directory, 1361,1300)

#GET NEAREST NEIGHBOURS USING PATCHES

patches_nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(y_patches_train)
distances, indices = patches_nbrs.kneighbors(y_patches_train)

get_similar_top_n(patches_nbrs,y_patches_test, patch_file_list[patches_split:], patch_file_list, patch_directory, 3)


patch_images_train = patch_images[:patches_split]
patch_images_train_files = patch_file_list[:patches_split]
patch_images_test = patch_images[patches_split:]
patch_images_test_files = patch_file_list[patches_split:]
patch_images_expanded_style = get_layer_vectors(patch_images_train, vgg_model.model)


style_dict = get_similar_top_style_wise(vgg_model.model, patch_images_test, patch_images_test_files, patch_directory, patch_images_train_files, patch_images_train)
