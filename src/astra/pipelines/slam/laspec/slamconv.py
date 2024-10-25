import tensorflow as tf
# from keras.models import Model
# from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense
# from keras.layers.core import Activation
# from keras.layers.core import Dropout
# from keras.layers.core import Lambda
# from keras.layers.core import Dense
# from keras.layers import Flatten
# from keras.layers import Input
# import tensorflow as tf
# from keras.utils import plot_model


def multi_model():
    inputs = tf.keras.layers.Input()
    inputs = Input(shape=(224, 224, 3), name='Input_image')
    ###################
    # Category Block ##
    ###################
    # utilize a lambda layer to convert the 3 channel input to a grayscale representation
    x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)
    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # (CONV => RELU) * 2 => POOL
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    category_out = Activation(activation="softmax", name="category_output")(x)
    ###################
    # Color Block #####
    ###################
    # CONV => RELU => POOL
    y = Conv2D(16, (3, 3), padding="same")(inputs)
    y = Activation("relu")(y)
    y = BatchNormalization(axis=-1)(y)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    y = Dropout(0.25)(y)

    # CONV => RELU => POOL
    y = Conv2D(32, (3, 3), padding="same")(y)
    y = Activation("relu")(y)
    y = BatchNormalization(axis=-1)(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Dropout(0.25)(y)

    # CONV => RELU => POOL
    y = Conv2D(32, (3, 3), padding="same")(y)
    y = Activation("relu")(y)
    y = BatchNormalization(axis=-1)(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Dropout(0.25)(y)

    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    y = Flatten()(y)
    y = Dense(128)(y)
    y = Activation("relu")(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Dense(15)(y)
    color_out = Activation(activation="softmax", name="color_output")(y)
    model = Model(inputs=inputs, outputs=[category_out, color_out])
    return model


model = multi_model()
model.summary()
model.compile(optimizer='sgd', loss=['categorical_crossentropy', 'categorical_crossentropy'])
plot_model(model, show_shapes=True, to_file='{}.png'.format("Keras Multi Branch"))