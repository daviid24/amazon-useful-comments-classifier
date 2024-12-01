from pathlib import Path
import string
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


class DatabaseLoader:
    def __init__(self, data_dir):
        """
        Inicializa el cargador de base de datos con los directorios de datos.

        Args:
            data_dir (str or Path): Directorio base donde se encuentran las carpetas `raw` y `processed`.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.df = None

        # Crear directorios si no existen
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, filename="data_full.csv"):
        """
        Carga la base de datos original desde el directorio `raw`.

        Args:
            filename (str): Nombre del archivo en el directorio `raw`.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        filepath = self.raw_dir / filename
        self.df = pd.read_csv(filepath, encoding="utf-8")
        self.df['product_category'] = self.df['product_category'].apply(lambda x: x.strip())
        return self.df

    @staticmethod
    def normalize_text(text):
        """
        Normaliza el texto: corrige caracteres mal codificados, convierte a minúsculas,
        elimina puntuación, números y stopwords.

        Args:
            text (str): Texto a normalizar.

        Returns:
            str: Texto normalizado.
        """
        # Corregir caracteres mal codificados
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

        # Convertir a minúsculas
        text = text.lower()

        # Eliminar signos de puntuación y números
        text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)

        # Tokenizar el texto
        tokens = nltk.word_tokenize(text)

        # Eliminar palabras de parada (stopwords)
        stop_words = set(stopwords.words('spanish'))  # Cambiar idioma si necesario
        tokens = [word for word in tokens if word not in stop_words]

        # Eliminar palabras muy cortas (ej: una letra)
        tokens = [word for word in tokens if len(word) > 1]

        # Unir de nuevo los tokens en una cadena
        normalized_text = " ".join(tokens)

        return normalized_text

    def normalize_reviews(self):
        """Aplica la normalización de texto a los comentarios."""
        self.df['normalized_review_body'] = self.df['review_body'].apply(self.normalize_text)

    @staticmethod
    def is_useful_comment(text, score, min_words=10):
        """
        Determina si un comentario es útil o no, basándose en la cantidad de palabras significativas y la puntuación.

        Args:
            text (str): Texto del comentario normalizado.
            score (int): Puntuación del comentario (1-5).
            min_words (int): Número mínimo de palabras significativas para considerar útil.

        Returns:
            str: 'useful' si el comentario es útil, 'not_useful' en caso contrario.
        """
        word_count = len(text.split())
        if word_count >= min_words and score in [1, 5]:
            return 'useful'
        else:
            return 'not_useful'

    def label_useful_comments(self):
        """Crea las columnas de etiquetas de utilidad para los comentarios."""
        self.df['useful_comment'] = self.df.apply(
            lambda row: self.is_useful_comment(row['normalized_review_body'], row['stars']), axis=1
        )
        self.df['useful_comment_flag'] = self.df['useful_comment'].apply(lambda x: 1 if x == 'useful' else 0)


    def save_processed_data(self, filename="comments_with_usefulness_flag.csv"):
        """
        Guarda la base de datos procesada en el directorio `processed`.

        Args:
            filename (str): Nombre del archivo a guardar.
        """
        filepath = self.processed_dir / filename
        self.df.to_csv(filepath, index=False)
        print(f"Base de datos procesada guardada en: {filepath}")

    def get_processed_data(self):
        """
        Retorna el DataFrame procesado.

        Returns:
            pd.DataFrame: DataFrame procesado con las etiquetas de utilidad.
        """
        return self.df