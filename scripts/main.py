from network import Network
from dataset_splitter import DatasetSplitter
from database_loader import DatabaseLoader
from tensorflow.keras.utils import to_categorical
import numpy as np


def main():
    # Definir el directorio base de los datos
    data_dir = "data"

    # Cargar y procesar la base de datos
    loader = DatabaseLoader(data_dir=data_dir)
    print("Cargando y procesando la base de datos...")
    loader.load_data()
    loader.normalize_reviews()
    loader.label_useful_comments()
    df_processed = loader.get_processed_data()

    # Prepara características y etiquetas
    print("Preparando datos para el modelo...")
    features = np.random.rand(len(df_processed), 20)  # Simulación; reemplaza con tus transformaciones reales
    target = df_processed['useful_comment_flag'].values

    # Dividir en conjuntos de entrenamiento, validación y prueba
    splitter = DatasetSplitter(features, target)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data()

    # Convertir etiquetas a formato one-hot
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    print(f"Conjuntos divididos: Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Prueba: {X_test.shape}")

    # Crear modelos con las nuevas configuraciones
    models = {
        "Basic": Network(input_shape=X_train.shape[1], layers=2, units=128),
        "Deep": Network(input_shape=X_train.shape[1], layers=4, units=128),
        "Dropout": Network(input_shape=X_train.shape[1], layers=4, units=128, dropout_rate=0.3),
        "BatchNorm": Network(input_shape=X_train.shape[1], layers=4, units=128, batch_norm=True),
        "Dropout + BatchNorm": Network(input_shape=X_train.shape[1], layers=4, units=128, dropout_rate=0.5, batch_norm=True),
        "Mixed": Network(input_shape=X_train.shape[1], layers=4, units=128, dropout_rate=0.3, regularization='l1'),
    }

    # Entrenar y evaluar cada modelo
    histories = {}
    results = {}
    for name, network in models.items():
        print(f"Entrenando modelo: {name}")
        histories[name] = network.train(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
        results[name] = network.evaluate(X_test, y_test)
        network.save_model(name)
        Network.plot_history(histories[name], name)

    # Comparar resultados
    for name, result in results.items():
        print(f"{name} - Pérdida: {result[0]}, Precisión: {result[1]}")

if __name__ == "__main__":
    main()