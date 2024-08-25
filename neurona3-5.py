import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class RegresionLineal(tf.keras.Model):
    """Modelo de regresión lineal implementado con APIs de alto nivel."""
    def __init__(self, num_entradas, lr):
        super(RegresionLineal, self).__init__()
        self.lr = lr
        inicializador = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=inicializador)
        self.optimizador = tf.keras.optimizers.SGD(learning_rate=self.lr)
    
    def call(self, X):
        return self.net(X)
    
    def calcular_perdida(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

# **Código actual con datos sintéticos**
# Generar datos sintéticos
w_verdadero = tf.constant([2, -3.4])
b_verdadero = 4.2
num_ejemplos = 1000

caracteristicas = tf.random.normal(shape=(num_ejemplos, len(w_verdadero)))
etiquetas = tf.tensordot(caracteristicas, w_verdadero, axes=1) + b_verdadero
etiquetas += tf.random.normal(shape=etiquetas.shape, stddev=0.01)

# Dividir datos en entrenamiento y validación
n_train = int(num_ejemplos * 0.8)
caracteristicas_entrenamiento = caracteristicas[:n_train]
etiquetas_entrenamiento = etiquetas[:n_train]
caracteristicas_validacion = caracteristicas[n_train:]
etiquetas_validacion = etiquetas[n_train:]

# **Código para cargar y usar un dataset propio**
# Descomentar estas líneas para cargar tu dataset personalizado
# df = pd.read_csv('ruta_al_dataset.csv')  # Ruta al archivo CSV de tu dataset

# Suponiendo que el dataset tiene múltiples características y una columna de etiquetas
# caracteristicas = df.drop('etiqueta', axis=1).values
# etiquetas = df['etiqueta'].values

# Dividir el dataset en entrenamiento y validación
# caracteristicas_entrenamiento, caracteristicas_validacion, etiquetas_entrenamiento, etiquetas_validacion = train_test_split(
#     caracteristicas, etiquetas, test_size=0.2, random_state=42)

# Crear el modelo
num_entradas = caracteristicas.shape[1]  # Número de características, se ajusta automáticamente
modelo = RegresionLineal(num_entradas=num_entradas, lr=0.03)

# Crear el dataset de entrenamiento
tamano_lote = 10
dataset_entrenamiento = tf.data.Dataset.from_tensor_slices((caracteristicas_entrenamiento, etiquetas_entrenamiento))
dataset_entrenamiento = dataset_entrenamiento.shuffle(buffer_size=len(caracteristicas_entrenamiento)).batch(tamano_lote)

# Variables para almacenar las pérdidas
perdidas_entrenamiento = []
perdidas_validacion = []

# Entrenamiento del modelo
num_epocas = 3
for epoca in range(num_epocas):
    perdida_total_entrenamiento = 0
    for X, y in dataset_entrenamiento:
        with tf.GradientTape() as cinta:
            y_hat = modelo(X)
            perdida = modelo.calcular_perdida(y_hat, y)
        gradientes = cinta.gradient(perdida, modelo.trainable_variables)
        modelo.optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        perdida_total_entrenamiento += perdida.numpy()
    
    perdida_media_entrenamiento = perdida_total_entrenamiento / len(dataset_entrenamiento)
    perdidas_entrenamiento.append(perdida_media_entrenamiento)
    
    # Calcular la pérdida de validación
    y_hat_validacion = modelo(caracteristicas_validacion)
    perdida_validacion = modelo.calcular_perdida(y_hat_validacion, etiquetas_validacion).numpy()
    perdidas_validacion.append(perdida_validacion)
    
    print(f'Época {epoca + 1}, pérdida de entrenamiento: {perdida_media_entrenamiento:.4f}, pérdida de validación: {perdida_validacion:.4f}')

# Obtener los pesos y el sesgo aprendidos
w, b = modelo.net.get_weights()
print(f'Pesos estimados: {w}')
print(f'Sesgo estimado: {b}')

# Crear una tabla con los resultados finales
resultados = pd.DataFrame({
    'Época': [num_epocas],
    'Pérdida de Entrenamiento': [perdidas_entrenamiento[-1]],  # Último valor de la pérdida de entrenamiento
    'Pérdida de Validación': [perdidas_validacion[-1]],  # Último valor de la pérdida de validación
    'Pesos Estimados': [w.flatten().tolist()],  # Convertir el tensor a una lista para mostrar en la tabla
    'Sesgo Estimado': [b[0]]  # Convertir el tensor a un valor escalar
})

# Visualizar los resultados
plt.figure(figsize=(6, 6))

# Visualizar la pérdida de entrenamiento y validación
plt.subplot(1, 1, 1)  # Solo un gráfico ocupando toda la ventana
plt.plot(range(1, num_epocas + 1), perdidas_entrenamiento, label='Pérdida de entrenamiento')
plt.plot(range(1, num_epocas + 1), perdidas_validacion, label='Pérdida de validación', linestyle='--')
plt.xlabel('Época', fontsize=14)
plt.ylabel('Pérdida', fontsize=14)
plt.title('Pérdida de Entrenamiento y Validación', fontsize=16)
plt.legend(fontsize=12)

# Mostrar la tabla con los resultados finales
plt.figure(figsize=(12, 3))  # Ajustar tamaño de la figura para la tabla
plt.axis('off')
tbl = plt.table(cellText=resultados.values,
                colLabels=resultados.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Ajustar el tamaño de la tabla y el tamaño del texto
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)  # Ajustar tamaño del texto
tbl.scale(1, 1)  # Ajustar tamaño de la tabla

plt.title('Resultados Finales', fontsize=16)
plt.show()
