# Import Densenet from Keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create the base pre-trained model
base_model = DenseNet121(weights='./models/nih/densenet.hdf5', include_top=False);

# Print the model summary
base_model.summary()

# Print out the first five layers
layers_l = base_model.layers

print("First 5 layers")
layers_l[0:5]


# Print out the last five layers
print("Last 5 layers")
layers_l[-6:-1]

# Get the convolutional layers and print the first 5
conv2D_layers = [layer for layer in base_model.layers 
                if str(type(layer)).find('Conv2D') > -1]
print("The first five conv2D layers")
conv2D_layers[0:5]

# Print out the total number of convolutional layers
print(f"There are {len(conv2D_layers)} convolutional layers")

# Print the number of channels in the input
print("The input has 3 channels")
base_model.input

# Print the number of output channels
print("The output has 1024 channels")
x = base_model.output
x

# Add a global spatial average pooling layer
x_pool = GlobalAveragePooling2D()(x)
x_pool

# Define a set of five class labels to use as an example
labels = ['Emphysema', 
          'Hernia', 
          'Mass', 
          'Pneumonia',  
          'Edema']
n_classes = len(labels)
print(f"In this example, you want your model to identify {n_classes} classes")

# Add a logistic layer the same size as the number of classes you're trying to predict
predictions = Dense(n_classes, activation="sigmoid")(x_pool)
print("Predictions have {n_classes} units, one for each class")
predictions

# Create an updated model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy')
# (You'll customize the loss function in the assignment!)

