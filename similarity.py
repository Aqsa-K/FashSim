from tensorflow.keras import backend as K
import sklearn
from sklearn import metrics
import tensorflow as tf
from skimage import data, io, filters
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.util import crop
import numpy as np
import os
import matplotlib.pyplot as plt


def get_layer_vectors(x, model):
    # x defines the input samples to pass through the network to get thier corresponding layer vectors
    
    layer_output_data = K.function([model.layers[0].input],
                                  [model.layers[1].output])
    layer_output = layer_output_data([x])[0]
    
    return layer_output


def get_top_similar(input_x, y, y_labels, n=10):
    # input_x is the input vector for which we have to get similar vectors (similar vectors to input x)
    # y is the list of vectors to compare agaisnt to find similar vectors from
    # n defines the number of top similar vectors to return
    
    similar_vecs_indices = []
    
    x = input_x.reshape(1,-1)
    cosine_dict = {}
    for i, vector in enumerate(y):
        v = vector.reshape(1,-1)
        cosine_sim = metrics.pairwise.cosine_similarity(x,v)
        cosine_dict[i] = cosine_sim[0][0]
    
    sorted_cosine_dict = sorted(cosine_dict.items(), key=lambda kv: kv[1], reverse=True)
    
    for i in range(n):
        idx = sorted_cosine_dict[i][0] # get indices of the vectors
        print(y_labels[idx])
        similar_vecs_indices.append(y_labels[idx])
        
    return similar_vecs_indices


def get_style_loss(content, target):
    return (np.square(content - target)).mean()

    
def compute_style_score(init_img_gram_layer, style_img_gram_layer):
    style_score = 0
#     weight_per_style_layer = 1.0 / float(params.num_style_layers)
    style_score += get_style_loss(init_img_gram_layer, style_img_gram_layer)
    
    return (1-style_score)
    


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_similar_top_n(nbrs, y_test, file_name_test, file_name_list, train_directory, n=1):
    distances, indices = nbrs.kneighbors(y_test, n)
    
    for i in range(len(file_name_test)):
        print(i)
        test_file_name = file_name_test[i]
        similar_file_name_list = [file_name_list[int(k)] for k in indices[i]]
        
        print("test_file_name", test_file_name)
        
        test_img_path = os.path.join(train_directory, test_file_name)
        test_img = data.imread(test_img_path)
        io.imshow(test_img)
        plt.show()
        
        for file_name in similar_file_name_list:
            print("sim file: ", file_name)
            sim_img_path = os.path.join(train_directory, file_name)
            sim_img = data.imread(sim_img_path)
            io.imshow(sim_img)
            plt.show()
        

# NEAREST NEIGHBORS

from sklearn.neighbors import NearestNeighbors
def nearest_neighbors(y_train, n=2):
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(y_train)
    distances, indices = nbrs.kneighbors(y_train)
    # indices
    
    return nbrs
    

def get_similar_top_style_wise(model, test_patch, patch_style_files_test, patch_directory, patch_style_files, patch_images_expanded):
    
    i = 0
    style_score_dict = {}
    for img in test_patch:

        print("test file : ", patch_style_files_test[i])
        img_path = os.path.join(patch_directory, patch_style_files_test[i])
        img = data.imread(img_path)
        io.imshow(img)
        plt.show()

        print(img.shape)
        img = resize(img, (224,224,3) , anti_aliasing=True)
        print(img.shape)
        img_expanded = tf.expand_dims(img, 0)
        img_style = get_layer_vectors(img_expanded, model)
        print("img style: ", img_style.shape)

        style_score_list = []
        for patch in patch_images_expanded:
            print(patch.shape)

            io.imshow(img)
            plt.show()

            patch_exp = tf.expand_dims(patch, 0)
            patch_style = get_layer_vectors(patch_exp, model)
            print(patch_style.shape)
            style_score = compute_style_score(img_style, patch_style)
            style_score_list.append(style_score)

        style_score_dict[str(i)]={}
        style_score_dict[str(i)]['style_score_list']=style_score_list
        max_style_score = max(style_score_list)
        max_idx = style_score_list.index(max_style_score)
        style_score_dict[str(i)]['max_style_score'] = max_style_score
        style_score_dict[str(i)]['max_style_score_idx'] = max_idx 

        print("sim file: ", patch_style_files[max_idx])
        img_path = os.path.join(patch_directory, patch_style_files[max_idx])
        img = data.imread(img_path)
        io.imshow(img)
        plt.show()


        i+=1
        
    return style_score_dict