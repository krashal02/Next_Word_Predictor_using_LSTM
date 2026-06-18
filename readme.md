# Next Word Predictor using LSTM

## Overview
This project builds a Next Word Prediction model using TensorFlow/Keras and an LSTM network.

The model is trained on a FAQ dataset and learns word sequences so it can predict the most likely next word given a sentence fragment.

Example:

Input:
```
The total duration of the course
```

Predicted Output:
```
The total duration of the course is ...
```

---

## Project Workflow

### 1. Data Preparation
- FAQ text is stored in a single string variable (`faqs`).
- Text is tokenized using Keras `Tokenizer`.
- Each sentence is converted into numerical tokens.

Example:

```
I love machine learning
```

becomes

```
[10, 25, 48, 90]
```

---

### 2. Sequence Generation

Training sequences are created incrementally.

Example:

```
I love machine learning
```

Generated sequences:

```
[I, love]
[I, love, machine]
[I, love, machine, learning]
```

This allows the model to learn next-word relationships.

---

### 3. Padding

Since all sequences have different lengths, they are padded to the same size.

```python
pad_sequences(..., padding="pre")
```

Maximum sequence length found in the dataset:

```python
max_len = 57
```

Input length used by the model:

```python
56
```

The last word is used as the target label.

---

### 4. Feature and Target Creation

```python
x = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
```

- `x` = input sequence
- `y` = next word

Target values are converted to one-hot vectors:

```python
to_categorical()
```

---

## Model Architecture

```python
model = Sequential()

model.add(Input(shape=(56,)))
model.add(Embedding(input_dim=283, output_dim=100))
model.add(LSTM(150))
model.add(Dense(283, activation="softmax"))
```

### Layer Details

#### Input Layer

```python
Input(shape=(56,))
```

Accepts sequences containing 56 tokens.

---

#### Embedding Layer

```python
Embedding(input_dim=283, output_dim=100)
```

Parameters:

- Vocabulary Size = 283
- Embedding Dimension = 100

Converts each word ID into a dense vector representation.

Output Shape:

```
(None, 56, 100)
```

---

#### LSTM Layer

```python
LSTM(150)
```

- 150 memory units
- Learns sequential patterns and context
- Captures relationships between previous words

Output Shape:

```
(None, 150)
```

---

#### Dense Layer

```python
Dense(283, activation="softmax")
```

Produces probabilities for every word in the vocabulary.

Output Shape:

```
(None, 283)
```

---

## Model Compilation

```python
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

### Loss Function
- Categorical Cross Entropy

### Optimizer
- Adam

### Metric
- Accuracy

---

## Training

```python
history = model.fit(
    x,
    y,
    epochs=100
)
```

The model is trained for 100 epochs.

---

## Prediction

```python
text = "The total duration of the course"
```

Prediction process:

1. Convert text to tokens.
2. Pad sequence to length 56.
3. Run model prediction.
4. Select the word with maximum probability.
5. Append predicted word to the sentence.
6. Repeat multiple times.

Code:

```python
pos = np.argmax(model.predict(padded_token_text))
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy

---

## Folder Structure

```
project/
│
├── next_word_predictor.ipynb
├── README.md
```

---

## Future Improvements

- Train on a larger text corpus
- Add dropout layers to reduce overfitting
- Use Bidirectional LSTM
- Use GRU or Transformer-based architectures
- Save and reload trained models
- Build a web application using Flask or Streamlit

---

## Learning Outcomes

This project demonstrates:

- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- One-hot encoding
- Embedding layers
- LSTM networks
- Multi-class classification
- Text generation

## 👨‍💻 Author

Krashal Yaduvanshi
