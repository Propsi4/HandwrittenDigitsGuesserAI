from ModelService import ModelService
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# Path: main.py

if __name__ == "__main__":
    modelService = ModelService()
    while True:
        # Menu
        print("1. Fit model")
        print("2. Predict")
        print("3. Save model")
        print("4. Load model")
        print("5. Test convolutional layers output")
        print("6. Display convolutional layers")
        print("9. Exit")
        choice = int(input("Enter your choice: "))
        match choice:
            case 1:
                try:
                    epochs = int(input("Enter number of epochs: "))
                    history = modelService.fit(epochs)
                    plt.plot(history.history['accuracy'])
                    plt.plot(history.history['val_accuracy'])
                    plt.title('Model accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc='upper left')
                    plt.show()
                except:
                    print("Error fitting model")
            case 2:
                try:
                    img_path = input("Enter path to image: ")
                    img = load_img(img_path, target_size=(28, 28), color_mode='grayscale')
                    img = img_to_array(img)
                    img = img.reshape(1, 28, 28, 1)
                    img = img.astype('float32')
                    img = img / 255.0
                    prediction = modelService.predict(img)
                    print("Predicted number is: " + str(np.argmax(prediction)))
                except:
                    print("Error predicting")
            case 3:
                try:
                    modelService.save()
                except:
                    print("Error saving model")
            case 4:
                try:
                    modelService.load()
                except:
                    print("Error loading model")
            case 5:
                try:
                    path = input("Enter path to image: ")
                    layer = int(input("Enter layer number: "))
                    modelService.display_layer_output(path, layer)
                except:
                    print("Error displaying layer output")
            case 6:
                try:
                    path = input("Enter path to image: ")
                    modelService.display_conv_layers(path)
                except:
                    print("Error displaying convolutional layers")
            case 9:
                exit()

    