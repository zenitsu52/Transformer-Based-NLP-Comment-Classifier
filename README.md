# Transformer-Based Multi-Task NLP Comment Classifier

This project implements a **Transformer-based Multi-Task Learning NLP model** that classifies a given text/comment into three different categories simultaneously:

1. **Emotion Classification** (6 classes)
2. **Violence Classification** (5 classes)
3. **Hate Speech Classification** (3 classes)

The model is built using **DistilBERT** and fine-tuned with a shared LSTM and individual task heads. It uses techniques such as **data augmentation**, **oversampling**, and **clean preprocessing** to improve performance.

---

## 🚀 Features

* **Multi-task Learning:** Predicts 3 labels in one forward pass.
* **Transformer Backbone:** Uses DistilBERT for contextual embeddings.
* **Data Augmentation (EDA):** Improves minority class learning for hate speech.
* **Clean Preprocessing:** Includes stopword removal and tokenization.
* **Custom Inference Pipeline:** Returns major category, sub-label, and confidence.
* **Supports Real-time Comment Classification.**

---

## 📁 Project Structure

```
MULTI TASK NLP/
│
├── data/
│   ├── emotions.csv
│   ├── hate_speech.csv
│   ├── violence.csv
│
└── multi_task_learning_nlp_final.ipynb
```

---

## 🧠 Model Architecture Overview

* **Shared Transformer Layer**: DistilBERT
* **Shared LSTM Layer** for sequence modeling
* **Task-Specific Heads:**

  * Emotion: Dense(6, softmax)
  * Violence: Dense(5, softmax)
  * Hate Speech: Dense(3, softmax)

---

## 📊 Datasets Used

### **1. Emotion Dataset**

* 6 emotion labels: sadness, joy, love, anger, fear, surprise

### **2. Violence Classification Dataset**

* Multi-class labels for violent/non-violent content

### **3. Hate Speech Dataset**

* Labels: hate speech, offensive language, neither
* Uses augmentation to balance classes

---

## 🛠️ Installation & Setup

1. Clone the repo (or download project folder)
2. Install dependencies:

```
pip install tensorflow transformers nltk numpy pandas scikit-learn matplotlib
```

3. Download NLTK data:

```
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

---

## 🏋️‍♂️ Training the Model

Training is performed inside the notebook using:

* Augmented hate data
* Stratified splits
* Multi-task fitting with 3 outputs

Use:

```
model.fit(train_inputs, train_labels, ...)
```

---

## 🔮 Inference Example

The project includes a `classify_text()` function that:

* Cleans the text
* Tokenizes with DistilBERT
* Runs prediction on all three heads
* Returns strongest head + label + confidence

Example usage:

```python
major, sub, conf = classify_text("I hate you so much!")
print(major, sub, conf)
```

Output:

```
Hate   hate_speech   0.94
```

---

## 📈 Model Performance

The model shows strong performance across all tasks:

* **Emotion Accuracy:** ~87%
* **Violence Accuracy:** ~99%
* **Hate Speech Accuracy:** improved significantly after augmentation (>87%)

Confusion matrices and classification reports are generated inside the notebook.

---

## 🛡️ Use Cases

* Social media moderation
* Comment filtering systems
* Toxicity detection
* Multi-label content analysis

---

## 📌 Future Improvements

* Use full BERT or RoBERTa for richer embeddings
* Add backtranslation for stronger augmentation
* Convert to a REST API or deploy with FastAPI
* Add explainability (LIME/SHAP)

---

## 📜 License

This project is open for educational and research purposes.

---

## 👨‍💻 Author

**Sahil Gawade**

Feel free to reach out for improvements or suggestions!
