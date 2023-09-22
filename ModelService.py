from model import model
import tensorflow as tf
import mnist
import matplotlib.pyplot as plt
from callbacks import callbacks
from config import save_path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
import numpy as np

class ModelService:
    def __init__(self, metrics=['accuracy']):
        self.model = model
        self.model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=metrics)
        self.x_train = mnist.train_images()
        self.y_train = mnist.train_labels()
        self.x_test = mnist.test_images()
        self.y_test = mnist.test_labels()
        self.model.summary()
    def get_model(self):
        return self.model

    def predict(self, input):
        return self.model.predict(input)

    def fit(self, epochs=5):
        return self.model.fit(self.x_train, self.y_train, epochs=epochs, callbacks=callbacks, batch_size=1024, validation_data=(self.x_test, self.y_test))

    def save(self):
        print("Saving weights to " + save_path)
        self.model.save(save_path)

    def load(self):
        print("Loading weights from " + save_path)
        self.model = tf.keras.models.load_model(save_path)
    
    def display_layer_output(self, path, layer=0):
        image = load_img(path, target_size=(28, 28), color_mode='grayscale')
        image = img_to_array(image)
        image_np = image.reshape(1, 28, 28, 1)
        # Display convolutional layers output
        layer_model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)

        feature_maps = layer_model(image_np)
        
        num_feature_maps = feature_maps.shape[3]
        square = int(num_feature_maps**0.5)  # Determine grid size based on the number of feature maps

        # Create a grid of subplots to display the feature maps
        fig, axes = plt.subplots(square, square, figsize=(8, 8))

        for i in range(square):
            for j in range(square):
                # Get the i * square + j-th feature map
                feature_map = feature_maps[0, :, :, i * square + j]
                # Set up the subplot
                axes[i, j].imshow(feature_map, cmap='viridis')  # Use 'viridis' colormap or adjust as needed
                axes[i, j].axis('off')

        plt.show()
    
    def display_conv_layers(self, path):
                # Define a new Model that will take an image as input, and will output
        # intermediate representations for all layers in the previous model
        successive_outputs = [layer.output for layer in self.model.layers]
        visualization_model = tf.keras.models.Model(inputs = self.model.input, outputs = successive_outputs)

        # Prepare a random input image from the training set.
        img = load_img(path, target_size=(28, 28), )  # this is a PIL image
        img = img.convert('L')
        x   = img_to_array(img)                           # Numpy array with shape (28, 28, 1)
        x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 28, 28, 1)
        # Scale by 1/255
        x /= 255.0
        # Run the image through the network, thus obtaining all
        # intermediate representations for this image.
        successive_feature_maps = visualization_model.predict(x)
                # These are the names of the layers, so you can have them as part of our plot
        layer_names = [layer.name for layer in model.layers]

        # Display the representations
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            
            if len(feature_map.shape) == 4:
                #-------------------------------------------
                # Just do this for the conv / maxpool layers, not the fully-connected layers
                #-------------------------------------------
                n_features = feature_map.shape[-1]  # number of features in the feature map
                size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
                
                # Tile the images in this matrix
                display_grid = np.zeros((size, size * n_features))
                
                #-------------------------------------------------
                # Postprocess the feature to be visually palatable
                #-------------------------------------------------
                for i in range(n_features):
                    x  = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *=  64
                    x += 128
                    x  = np.clip(x, 0, 255).astype('uint8')
                    display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

                #-----------------
                # Display the grid
                #-----------------
                scale = 20. / n_features
                plt.figure( figsize=(scale * n_features, scale) )
                plt.title ( layer_name )
                plt.grid  ( False )
                plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
        plt.show()
        

