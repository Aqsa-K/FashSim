from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

# Class for loading vgg model and defining vgg loss

class VGG_MODEL():
  
    def __init__(self, image_shape):

        # Create vgg model
        self.image_shape = image_shape
        self.model = self.vgg_model()

    def vgg_model(self):

        vgg19 = VGG19(weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        #Set trainable for all layers as False
        for l in vgg19.layers:
            l.trainable = False

        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)

        return model



    def vgg_loss(self,y_true, y_pred):
        # Compute loss as the mean of square difference between true and predicted values
        loss = K.mean(K.square(self.model(y_true)-self.model(y_pred)))

        return loss


    def get_optimizer(self):
        # Optimizer Adam for tweaking parameters and reducing loss
        adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        return adam