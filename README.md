# IMDB Movie Review Rating Prediction — Artificial Neural Network (ANN) Project

## 1. Project Overview

This project is a **multi-class text classification** problem that aims to predict the **rating (1-10)** a user gave to a movie based on the text of their IMDB review. The solution uses a classic NLP preprocessing pipeline (TF-IDF based) combined with an **Artificial Neural Network (ANN)** built from fully connected (Dense) layers in Keras/TensorFlow.

- **Code source:** A notebook downloaded directly from Kaggle and run as-is.
- **Dataset source:** [IMDB Review Dataset — Kaggle](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset)
- **Environment:** Kaggle Notebook (uses `/kaggle/input/` and `/kaggle/working/` paths).
- **Main libraries:** `pandas`, `numpy`, `nltk`, `scikit-learn`, `tensorflow.keras`, `matplotlib`, `seaborn`.

---

## 2. Dataset

### 2.1 Source
The dataset comes from [ebiswas/imdb-review-dataset](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset) on Kaggle. It contains millions of IMDB movie reviews, including fields such as review text, rating, movie title, reviewer, and date, split into JSON chunk files (`part-00.json`, `part-01.json`, ...).

Only the **`part-04.json`** file is used in this project:

```python
with open('/kaggle/input/imdb-review-dataset/part-04.json', 'r') as file:
    data = json.load(file)
```

### 2.2 Raw Data Columns
The DataFrame built from the JSON contains the following columns:

```
['review_id', 'reviewer', 'movie', 'rating', 'review_summary',
 'review_date', 'spoiler_tag', 'review_detail', 'helpful']
```

Only **`rating`** (target label) and **`review_detail`** (review text, input feature) are used in this project; the remaining columns (`review_id`, `reviewer`, `movie`, `review_summary`, `review_date`, `spoiler_tag`, `helpful`) are dropped.

### 2.3 Data Cleaning
- Rows with missing (`NaN`) values are dropped using `dropna()`.
- After `dropna()`, `part-04.json` contains roughly **1,019,000 rows**.

### 2.4 Class Distribution (Imbalance Problem)
The raw dataset's rating distribution is heavily imbalanced; 10-star reviews (~211,000) are by far the most common, while 2-star reviews (~37,000) are the least common:

| Rating | Number of Reviews |
|--------|--------------------|
| 10   | 211,184 |
| 8    | 123,797 |
| 9    | 107,059 |
| 7    | 101,196 |
| 1    | 86,258  |
| 6    | 73,236  |
| 5    | 55,337  |
| 4    | 42,383  |
| 3    | 39,650  |
| 2    | 37,653  |

**Solution — Class Balancing:** A random sample of **10,000** reviews is taken from each rating class (`groupby('rating').sample(n=10000, random_state=63)`), producing a balanced subset (`ds`) of **100,000 rows total** with an equal number of examples per class. This is a standard approach to prevent the model from becoming biased toward majority classes (e.g., 10-star reviews).

Labels are also shifted to the `0-9` range (`rating - 1`), since Keras's `SparseCategoricalCrossentropy` loss expects 0-indexed integer labels.

---

## 3. Text Preprocessing (NLP Pipeline)

The following steps are applied to the review text (`review_detail`) in order:

1. **Lowercasing:** `str.lower()`
2. **Removing punctuation:** `re.sub(r'[^\w\s]', '', x)`
3. **Removing digits:** `re.sub(r'\d+', '', x)`
4. **Tokenization (splitting into words):** `str.split()`
5. **Stopword removal:** Using NLTK's English `stopwords` list (removing words like "the", "is", "and", etc.).
6. **Lemmatization:** Using `nltk.WordNetLemmatizer` to reduce words to their dictionary base form (e.g., "movies" → "movie").

After these steps, a cleaned, single-string `lemmatized_tokens` column is produced for each review.

> **Note:** NLTK's `stopwords`, `wordnet`, and `omw-1.4` data packages are downloaded via `nltk.download()`; the WordNet zip file is additionally extracted manually with `unzip` (a Kaggle-environment-specific requirement).

---

## 4. Feature Extraction: TF-IDF Vectorization

The cleaned text is converted into numeric vectors using **TF-IDF (Term Frequency – Inverse Document Frequency)**:

```python
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=10, max_df=0.9)
tfidf_matrix = tfidf_vectorizer.fit_transform(ds["lemmatized_tokens"])
```

Meaning of the parameters:
- **`max_features=1000`:** Limits the vocabulary to the 1000 most important words (by TF-IDF score) — controls the input dimensionality (and thus complexity) of the model.
- **`min_df=10`:** Ignores words that appear in fewer than 10 documents (filters out rare/noise words).
- **`max_df=0.9`:** Ignores words that appear in more than 90% of documents (filters out very common, non-discriminative words).

The result is a **100,000 × 1,000** dimensional `tfidf_df` matrix, where each row represents a review and each column represents a word's TF-IDF score. This matrix is used as the input to the neural network.

---

## 5. Model Architecture: Artificial Neural Network (ANN)

The model is built using Keras's **Functional API** as a classic feed-forward ANN made entirely of **Dense (fully connected) layers**:

```python
inputs = Input(shape=(tfidf_df.shape[1],))          # 1000-dimensional TF-IDF input

hidden = Dense(256, activation="relu")(inputs)
hidden = Dropout(0.3)(hidden)

hidden = Dense(128, activation="relu")(hidden)
hidden = Dropout(0.3)(hidden)

hidden = Dense(64, activation="relu")(hidden)
hidden = Dropout(0.3)(hidden)

hidden = Dense(32, activation="relu")(hidden)
hidden = Dropout(0.3)(hidden)

outputs = Dense(10, activation="softmax", name="output_layer")(hidden)

model = Model(inputs=inputs, outputs=outputs)
```

### Architecture Summary

| Layer | Neurons | Activation | Purpose |
|--------|-------------|------------|------|
| Input | 1000 | – | TF-IDF vector input |
| Dense 1 | 256 | ReLU | Feature learning |
| Dropout 1 | – | – | Prevents overfitting, rate: 0.3 |
| Dense 2 | 128 | ReLU | Feature learning |
| Dropout 2 | – | – | Rate: 0.3 |
| Dense 3 | 64 | ReLU | Feature learning |
| Dropout 3 | – | – | Rate: 0.3 |
| Dense 4 | 32 | ReLU | Feature learning |
| Dropout 4 | – | – | Rate: 0.3 |
| Output | 10 | Softmax | Probability distribution over 10 classes (ratings 1-10) |

**Total parameters:** 299,818 (approx. 1.14 MB)

### Compile Settings
```python
optimizer = Adam(learning_rate=0.0001)
loss = SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
```
- **Optimizer:** Adam, with a low learning rate (0.0001) — chosen for stable but slow convergence.
- **Loss:** `SparseCategoricalCrossentropy` — used because labels are integer-encoded (0-9) rather than one-hot encoded.
- **Metric:** Accuracy.

---

## 6. Train/Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    tfidf_df, ds["rating"], test_size=0.2, random_state=48
)
```
- The data is split into **80% train / 20% test** (80,000 training and 20,000 test examples out of 100,000).
- The label distributions of the train and test sets are visualized with histograms to confirm they remained balanced.
- Both the training and test sets are shuffled using `shuffle()`.

### Training Hyperparameters
```python
epochs = 200
batch_size = 32
```
The model is trained with `model.fit(x_train, y_train, epochs=200, batch_size=32)`.

> **Note:** The notebook output only shows the first few epochs (1-4); training accuracy is recorded at ~11.2% for epoch 1 and ~25.2% by epoch 4. The full 200-epoch training run appears truncated in the notebook's saved output.

---

## 7. Results and Evaluation

### 7.1 Test Set Performance
```python
model.evaluate(x_test, y_test, batch_size=32)
# Loss: 5.46
# Accuracy: ~23.0%
```

Raw accuracy is around **23%**. While this is noticeably above the random-guess baseline (10%) for a 10-class problem, it is low in absolute terms. The main reasons for this include:
- Ratings from 1-10 are inherently an **ordinal** problem, whereas the model treats it as an unordered (nominal) multi-class problem. For example, predicting 7 when the true rating is 8 is penalized the same as predicting 1, from the perspective of loss/accuracy.
- Only TF-IDF (a "bag-of-words" representation that ignores word order and context) is used as the feature representation; no more advanced representations (word embeddings, transformer-based models, etc.) are employed.
- Textual differences between neighboring ratings (e.g., 7 vs. 8) are often quite subtle.

### 7.2 Confusion Matrix
Predictions on the test set are compared against the true labels to plot a confusion matrix (`seaborn.heatmap`). This matrix visually shows which ratings the model tends to confuse with which other ratings.

### 7.3 "Closeness"-Based Evaluation
Recognizing that standard accuracy is a strict metric for ordinal data, two additional metrics are computed:

```python
difference = np.abs(prediction_list - true_value_list)
close_elements = difference <= 1
print(count_close.sum() / len(true_value_list))   # ≈ 0.576

weights = 1 / (difference + 1)
print(weights.sum() / len(true_value_list))        # ≈ 0.515
```

- **Accuracy within ±1 point:** **57.6%** of predictions are within 1 unit of the true rating (i.e., the model is "approximately correct" most of the time).
- **Weighted score (1 / (difference + 1)):** Average of **51.5%** — the further a prediction is from the true value, the lower its weighted contribution.

This shows that although the model's exact-match accuracy is low, its predictions are mostly **close** to the true rating — meaning the model captures the review's overall positive/negative sentiment reasonably well, even when it misses the exact score.

---

## 8. End-to-End Pipeline Summary

```
1. Data Loading         → part-04.json (Kaggle IMDB Review Dataset)
2. Column Selection     → rating, review_detail
3. Cleaning             → dropna()
4. Class Balancing      → 10,000 samples per rating (100,000 total)
5. Text Preprocessing   → lowercasing, punctuation/digit removal,
                          tokenization, stopword removal, lemmatization
6. Feature Extraction   → TF-IDF (max_features=1000)
7. Train/Test Split     → 80% / 20%
8. Model                → 4 hidden Dense layers + Dropout(0.3), Softmax output
9. Training              → Adam(lr=0.0001), 200 epochs, batch_size=32
10. Evaluation           → Accuracy (~23%), Confusion Matrix,
                          ±1 closeness ratio (~57.6%)
```

---

## 9. Technologies Used

| Category | Tool/Library |
|----------|----------------|
| Data processing | pandas, numpy |
| Text preprocessing | nltk (stopwords, WordNet lemmatizer), re (regex) |
| Feature extraction | scikit-learn (`TfidfVectorizer`) |
| Model / Deep Learning | TensorFlow / Keras (Functional API, Dense, Dropout, Adam) |
| Evaluation | scikit-learn (`confusion_matrix`), `train_test_split` |
| Visualization | matplotlib, seaborn |

---

## 10. Potential Improvement Ideas

This section is not part of the original notebook, but is worth adding to a project report:

- **Ordinal regression approach:** Treating the problem as regression (single output neuron, MSE/MAE loss) instead of classification could better capture the ordinal relationship between ratings.
- **Richer text representations:** Using word embeddings (Word2Vec, GloVe) or pretrained transformer models (BERT, DistilBERT) instead of TF-IDF.
- **Sequence models:** Incorporating word order into the model using architectures such as LSTM/GRU or 1D-CNN instead of plain Dense layers.
- **Early stopping:** Monitoring validation loss to prevent overfitting instead of training for a fixed 200 epochs.
- **Learning rate scheduling:** Trying a decaying learning rate schedule instead of a fixed 0.0001.

---

## 11. References

- Dataset: [IMDB Review Dataset — Kaggle (ebiswas)](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset)
- Code: Notebook downloaded from Kaggle (`imdb-ann-project.ipynb`)