import tensorflow as tf
import matplotlib.pyplot as plt

class Clasificador:
    def __init__(self):
        self.hparams = None
    
    def guardar_hiperparametros(self, **kwargs):
        self.hparams = kwargs

class MLPScratch(Clasificador):
    def __init__(self, num_entradas, num_salidas, num_ocultas, lr, sigma=0.01):
        super().__init__()
        self.guardar_hiperparametros(num_entradas=num_entradas, num_salidas=num_salidas, num_ocultas=num_ocultas, lr=lr, sigma=sigma)
        self.W1 = tf.Variable(tf.random.normal((num_entradas, num_ocultas)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_ocultas))
        self.W2 = tf.Variable(tf.random.normal((num_ocultas, num_salidas)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_salidas))

    @property
    def variables_entrenables(self):
        return [self.W1, self.b1, self.W2, self.b2]

def relu(X):
    return tf.math.maximum(X, 0)

class MLPScratch(MLPScratch):
    def forward(self, X):
        X = tf.reshape(X, (-1, self.hparams['num_entradas']))
        H = relu(tf.matmul(X, self.W1) + self.b1)
        return tf.matmul(H, self.W2) + self.b2

def cargar_datos_fashion_mnist(tamano_lote, resize=None):
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    procesar = lambda X, y: (tf.cast(tf.expand_dims(X, axis=-1), tf.float32) / 255.0, tf.cast(y, dtype=tf.int32))
    iter_train = tf.data.Dataset.from_tensor_slices(procesar(*mnist_train)).shuffle(60000).batch(tamano_lote)
    iter_test = tf.data.Dataset.from_tensor_slices(procesar(*mnist_test)).batch(tamano_lote)
    return iter_train, iter_test

class Entrenador:
    def __init__(self, max_epocas):
        self.max_epocas = max_epocas
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def ajustar(self, modelo, datos):
        optimizador = tf.keras.optimizers.SGD(modelo.hparams['lr'])
        perdida = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        iter_train, iter_test = datos

        for epoca in range(self.max_epocas):
            # Entrenamiento
            epoch_loss_avg = tf.keras.metrics.Mean()
            for X, y in iter_train:
                with tf.GradientTape() as tape:
                    y_hat = modelo.forward(X)
                    l = perdida(y, y_hat)
                gradientes = tape.gradient(l, modelo.variables_entrenables)
                optimizador.apply_gradients(zip(gradientes, modelo.variables_entrenables))
                epoch_loss_avg.update_state(l)
            self.history['train_loss'].append(epoch_loss_avg.result().numpy())

            # Validación
            epoch_val_loss_avg = tf.keras.metrics.Mean()
            epoch_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            for X_val, y_val in iter_test:
                val_logits = modelo.forward(X_val)
                val_loss = perdida(y_val, val_logits)
                epoch_val_loss_avg.update_state(val_loss)
                epoch_val_accuracy.update_state(y_val, val_logits)
            self.history['val_loss'].append(epoch_val_loss_avg.result().numpy())
            self.history['val_acc'].append(epoch_val_accuracy.result().numpy())

            print(f'Época {epoca + 1}, Pérdida {epoch_loss_avg.result().numpy():.3f}, Pérdida val. {epoch_val_loss_avg.result().numpy():.3f}, Prec. val. {epoch_val_accuracy.result().numpy():.3f}')

    def plot_loss(self):
        epochs = range(1, self.max_epocas + 1)
        plt.plot(epochs, self.history['train_loss'], label='Pérdida entrenamiento')
        plt.plot(epochs, self.history['val_loss'], label='Pérdida validación', linestyle='--')
        plt.plot(epochs, self.history['val_acc'], label='Precisión validación', linestyle='-.')
        plt.title('Pérdida y precisión durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida / Precisión')
        plt.legend()
        plt.show()

class MLP(Clasificador):
    def __init__(self, num_salidas, num_ocultas, lr):
        super().__init__()
        self.guardar_hiperparametros(num_salidas=num_salidas, num_ocultas=num_ocultas, lr=lr)
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_ocultas, activation='relu'),
            tf.keras.layers.Dense(num_salidas)
        ])

    @property
    def variables_entrenables(self):
        return self.net.trainable_variables

    def forward(self, X):
        return self.net(X)

tamano_lote = 100
datos = cargar_datos_fashion_mnist(tamano_lote=tamano_lote)

# Entrenar MLPScratch
modelo_scratch = MLPScratch(num_entradas=784, num_salidas=10, num_ocultas=256, lr=0.1)
entrenador_scratch = Entrenador(max_epocas=10)
entrenador_scratch.ajustar(modelo_scratch, datos)
entrenador_scratch.plot_loss()

# Entrenar MLP
modelo = MLP(num_salidas=10, num_ocultas=256, lr=0.1)
entrenador = Entrenador(max_epocas=10)
entrenador.ajustar(modelo, datos)
entrenador.plot_loss()
