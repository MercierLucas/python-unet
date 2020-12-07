import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam


class Unet:
    # In this architecture convolutions are by blocks of 2
    def doubleConv(self,inputLayer,n_filters=16,kernel_size=3,stride=1,padding="same",batchnorm=True):
        x = layers.Conv2D(
            n_filters,
            (kernel_size,kernel_size),
            activation='relu',
            strides = (stride,stride),
            padding="same")(inputLayer)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            n_filters,
            (kernel_size,kernel_size),
            activation='relu',
            strides = (stride,stride),
            padding="same")(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        return x
    
    def __init__(self,input_size,lr=0.001,activation="relu",loss="binary_crossentropy",metric="accuracy",filter_size=16,step_size=5):
        input_img = tf.keras.Input((input_size,input_size,1),name="img")
        self.model = self.getModel(input_img,activation=activation)
        self.compileModel(loss=loss,metric=metric,lr=lr)
        
    def compileModel(self,loss="binary_crossentropy",metric="accuracy",lr=0.0001):
        print("Compiling model with lr {}...".format(lr))
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            #run_eagerly=True
            metrics=[metric]
        )
        #self.model.summary()
        print("Compilation done. The model can now be trained.")
        
    def train(self,X_train,Y_train,X_test,Y_test,epochs=5,batch_size=32):
        self.results = self.model.fit(
            X_train,Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test,Y_test)
        )
        return self.results
    
    def predict(self,X):
        return self.model.predict(X)
    
    def plotLearningCurve(self):
        #code from @hlamba28 /UNET-TGS 
        plt.figure(figsize=(8, 8))
        plt.title("Learning curve")
        plt.plot(self.results.history["loss"], label="loss")
        plt.plot(self.results.history["val_loss"], label="val_loss")
        plt.plot( np.argmin(self.results.history["val_loss"]), np.min(self.results.history["val_loss"]), marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend();
        
    def getModel(self,input_img,activation="relu",filter_size=16,step_size=5,dropout=0.1):
        convs = []
        pool = []
        encoding_steps = [2**i for i in range(step_size)]
        print("Generating model with 2@{} convultional layers : {}".format(step_size,[i*filter_size for i in encoding_steps]))
        last_tensor = input_img

        #encoder
        for i in range(len(encoding_steps)):
            c = self.doubleConv(last_tensor,n_filters = filter_size*encoding_steps[i],kernel_size = 3)

            if i != len(encoding_steps)-1:
                convs.append(c)
                last_tensor = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(c)
                last_tensor = layers.Dropout(dropout)(last_tensor)
            else:
                last_tensor = c

        decoding_steps = encoding_steps[:len(encoding_steps)-1]
        decoding_steps.reverse()
        size_i = i+1
        #decoder
        for i in range(len(decoding_steps)):
            u = layers.Conv2DTranspose(filter_size * decoding_steps[i],(3,3),strides = (2,2),padding="same")(last_tensor)
            u = tf.keras.layers.Concatenate()([u,convs[len(convs)-(i+1)]])
            u = layers.Dropout(dropout)(u)
            last_tensor = self.doubleConv(u,filter_size * decoding_steps[i],3)
            

        outputs = layers.Conv2D(1,(1,1),activation=activation)(last_tensor)
        
        print("Model generated")
        return models.Model(inputs=[input_img],outputs=[outputs])
