import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from pathlib import Path


class Network:
    def __init__(self, input_shape, layers=2, units=128, dropout_rate=0.0, batch_norm=False, regularization=None):
        """
        Inicializa una red neuronal configurable.

        Args:
            input_shape (int): Número de características de entrada.
            layers (int): Número de capas ocultas.
            units (int): Número de unidades por capa oculta.
            dropout_rate (float): Tasa de Dropout.
            batch_norm (bool): Si se utiliza Batch Normalization.
            regularization (str): Tipo de regularización ('l1', 'l2' o None).
        """
        self.input_shape = input_shape
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.regularization = regularization
        self.model = self._build_model()

    def _build_model(self):
        """Construye la red neuronal basada en la configuración."""
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))

        reg = None
        if self.regularization == 'l1':
            reg = l1(0.01)
        elif self.regularization == 'l2':
            reg = l2(0.01)

        for _ in range(self.layers):
            model.add(Dense(self.units, activation='relu', kernel_regularizer=reg))
            if self.batch_norm:
                model.add(BatchNormalization())
            if self.dropout_rate > 0.0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(2, activation='softmax'))  # Clasificación binaria
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, validation_data=None, epochs=10, batch_size=32):
        """
        Entrena la red neuronal.

        Args:
            X_train (np.array): Conjunto de entrenamiento (características).
            y_train (np.array): Conjunto de entrenamiento (etiquetas).
            validation_data (tuple): Tupla con los datos de validación.
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del batch.

        Returns:
            history: Objeto de historia del entrenamiento.
        """
        history = self.model.fit(X_train, y_train, validation_data=validation_data,
                                 epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en un conjunto de prueba.

        Args:
            X_test (np.array): Conjunto de prueba (características).
            y_test (np.array): Conjunto de prueba (etiquetas).

        Returns:
            tuple: Pérdida y exactitud en el conjunto de prueba.
        """
        return self.model.evaluate(X_test, y_test, verbose=1)

    @staticmethod
    def plot_history(history, model_name):
        """
        Grafica la pérdida y precisión del entrenamiento y validación.

        Args:
            history: Objeto de historia del entrenamiento.
            model_name (str): Nombre del modelo para el título de las gráficas.
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        # Precisión
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Entrenamiento')
        plt.plot(epochs, val_acc, 'ro-', label='Validación')
        plt.title(f'Precisión - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()

        # Pérdida
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Entrenamiento')
        plt.plot(epochs, val_loss, 'ro-', label='Validación')
        plt.title(f'Pérdida - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()

        plt.show()

    def save_model(self, model_name, models_dir="models"):
        """
        Guarda el modelo en formato .h5 en la carpeta `models`.

        Args:
            model_name (str): Nombre del modelo a guardar.
            models_dir (str): Carpeta para guardar el modelo (por defecto "models").
        """
        models_dir_path = Path(models_dir)
        models_dir_path.mkdir(parents=True, exist_ok=True)
        model_path = models_dir_path / f"{model_name}.h5"
        self.model.save(model_path)
        print(f"Modelo guardado en: {model_path}")