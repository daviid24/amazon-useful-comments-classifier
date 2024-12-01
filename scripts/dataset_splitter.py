from sklearn.model_selection import train_test_split


class DatasetSplitter:
    def __init__(self, features, target):
        """
        Inicializa el divisor de datos.

        Args:
            features (np.array): Matriz de características (X).
            target (np.array): Etiquetas de salida (y).
        """
        self.features = features
        self.target = target

    def split_data(self, test_size=0.5, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento, prueba y validación.

        Args:
            test_size (float): Tamaño del conjunto de prueba respecto a los datos totales.
            random_state (int): Semilla para la reproducibilidad.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test