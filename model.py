import tensorflow.keras as keras

model = keras.models.Sequential([
    # keras.layers.Conv2D(32, (3, 3),input_shape=(28, 28, 1), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(16, (3, 3),input_shape=(28,28,1) , activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (2, 2), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])