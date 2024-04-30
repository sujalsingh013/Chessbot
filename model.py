import tensorflow as tf
import numpy as np
class attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True)
    def call(self, inputs):
        attention_weights = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3),padding = 'same', activation='relu', input_shape=(1,8,8)),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding = 'same'),
        tf.keras.layers.Flatten(),
        ])
        self.lstm_model=tf.keras.Sequential([
        attention(),
        tf.keras.layers.LSTM(128, return_sequences=True),
        attention(),
        tf.keras.layers.LSTM(256, dropout=0.2, return_sequences=True),
        tf.keras.layers.Dense(256, activation='relu'),])
    def call(self, input):
        input1=input[0]
        input2=input[1]
        input1=np.array(input1)
        input1=tf.convert_to_tensor(input1)
        input2=np.array(input2)
        input2=tf.convert_to_tensor(input2)
        x = self.cnn_model(input1)
        y = self.lstm_model(input2)
        combined_output = tf.keras.layers.concatenate([x, y])

        # Add additional layers for further processing
        x = tf.keras.layers.Dense(256, activation='relu')(combined_output)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        # Output layer for Q value
        output = tf.keras.layers.Dense(1, activation='relu')(x)
        return output