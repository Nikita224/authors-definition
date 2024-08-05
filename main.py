import sqlite3                                                                         # для подключения и взаимодействия с базой данных SQLite
import pandas as pd                                                                    # мощные структуры данных и инструменты для анализа данных
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer          # CountVectorizer преобразует текст в матрицу счетчиков токенов. TfidfTransformer преобразует матрицу счетчиков токенов в матрицу TF-IDF
from sklearn.naive_bayes import MultinomialNB                                          # наивный байесовский классификатор для многономиальных распределений
from sklearn.pipeline import Pipeline                                                  # создание конвейеров обработки данных и построения моделей
from sklearn.model_selection import train_test_split                                   # разделение данных на обучающую и тестовую выборки (не используется)
from sklearn.preprocessing import LabelEncoder                                         # преобразование категориальных меток в числовые значения
import re                                                                              # модуль для работы с регулярными выражениями, используется для очистки текста
import nltk                                                                            # библиотека для обработки естественного языка
from nltk.corpus import stopwords                                                      # Список стоп-слов из корпуса NLTK
from nltk.stem import WordNetLemmatizer                                                # лемматизатор из WordNet

# Загрузка ресурсов NLTK
nltk.download('stopwords') # пример стоп слов ["the", "is", "in", "and", "to", "with"]
nltk.download('punkt')     # пунктуационный токенизатор
nltk.download('wordnet')   # привежение к нормальной форме

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower() # все в нижний регистр
    text = re.sub(r'<[^>]+>|[^\w\s]', '', text)  # удаление спецсимволов
    text = re.sub(r'\d+', '', text)  # удаление цифр
    text = re.sub(r'\s+', ' ', text).strip()  # удаление лишних пробелов
    tokens = nltk.word_tokenize(text) # токенизация на слова
    tokens = [word for word in tokens if word not in stopwords.words('english')] # поиск и удаление частых слов и слов, не несущих смысловой нагрузки 
    lemmatizer = WordNetLemmatizer() # создание лемантизайера
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # приведение к нормальной форме с помощью лемантизайера
    return ' '.join(tokens) # возврат строки с токенами через пробел

# Подключение к базе данных и загрузка данных
db_path = 'db/PubMedArticles-7.db' # ссылка на БД SQLite
conn = sqlite3.connect(db_path) # подключение к БД
query = "SELECT Author, Abstract FROM ArticleStruct" # выбор 2-ух полей (без заголовка текста)
data = pd.read_sql_query(query, conn) # выполнение запроса
conn.close() # закрытие соединения

# Предобработка данных
data.dropna(subset=['Author', 'Abstract'], inplace=True) # удаление из загруженных данных неполных данных
data['Abstract'] = data['Abstract'].apply(preprocess_text) # процесс подготовки данных 

# Кодирование меток (авторов)
label_encoder = LabelEncoder() # создание экземпляра класса для преобразования категориальных данных (текста) в числовые метки
data['Author'] = label_encoder.fit_transform(data['Author']) # преобразование каждого уникального значения в уникальное числовое значение

# Разделение данных на обучающую и тестовую выборки
X = data['Abstract'].values # X хранит массив текстов авторов
y = data['Author'].values   # Y хранит массив кодов авторов
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) применимо только если бы было несколько статей с этими же авторами

# Создание пайплайна для обработки текста и обучения наивного байесовского классификатора.
text_clf = Pipeline([
    ('vect', CountVectorizer(max_features=10000)),  # инструмент для преобразования текста в матрицу счетчиков токенов.
                                                    # преобразует коллекцию текстовых документов в матрицу, где каждый элемент (слово) представляется количеством его вхождений в документ
                                                    # ограничивает количество уникальных слов (токенов), которые будут учитываться. В данном случае, это 10000 наиболее часто встречающихся слов.
    
    ('tfidf', TfidfTransformer()),                  # инструмент для преобразования матрицы счетчиков токенов в матрицу TF-IDF
                                                    # TF-IDF — это статистическая мера, используемая для оценки важности слова в контексте документа и набора документов
                                                    # помогает уменьшить влияние часто встречающихся слов, которые могут быть менее информативными

    ('clf', MultinomialNB()),                       # наивный байесовский классификатор для многономиальных распределений
])

# Обучение модели
text_clf.fit(X, y) # тренировка модели машинного обучения

# Функция для предсказания автора текста
def predict_author(text, model, label_encoder):
    text = preprocess_text(text)    # тест с консоли обрабатываем
    prediction = model.predict([text])  # предсказание модели автора. Возвращает числовую метку
    author = label_encoder.inverse_transform(prediction) # расшифровываем метку автора в текст автора
    return author[0]    # возвращаем предсказанного автора

# Интерфейс для ввода текста и предсказания
while True:
    text_to_predict = input("Введите текст для предсказания автора (или 'exit' для выхода): ")
    if text_to_predict.lower() == 'exit':
        break
    predicted_author = predict_author(text_to_predict, text_clf, label_encoder)
    print(f"Predicted author: {predicted_author}")
