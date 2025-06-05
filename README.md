# news-category-rnn
# 📰 News Category Classification using LSTM

This project classifies news headlines into one of 10 categories using a Long short term memory (LSTM). It demonstrates how to clean text data, tokenize it, train an LSTM model, and build a web app using Streamlit.

---

## 🚀 Project Structure

- `explore_data.ipynb` – Explore and visualize the dataset.
- `train_model.ipynb` – Text cleaning, tokenization, and training an LSTM model.
- `evaluate_model.ipynb` – Evaluate model accuracy and predictions.
- `streamlit_app.py` – Streamlit app for predicting news categories.

---

## 🧠 Model Information

- **Architecture**: LSTM with Keras Sequential API
- **Embedding Layer**: 128-dimensional word vectors
- **Tokenizer**: Keras Tokenizer with `<OOV>` token
- **Output Classes**: 10 News Categories
- **Loss Function**: Categorical Crossentropy
- **Activation Function**: Softmax

---

## 📊 Dataset

This project uses the [News Category Dataset](https://arxiv.org/abs/2209.11429) created by **Rishabh Misra**.

If you are using this dataset, please cite:

> 1. Misra, Rishabh. "News Category Dataset." *arXiv preprint* arXiv:2209.11429 (2022).  
> 2. Misra, Rishabh and Jigyasa Grover. *Sculpting Data for ML: The first act of Machine Learning.* ISBN 9798585463570 (2021).

---

## 📥 Categories Predicted

The model classifies news headlines into the following 10 categories:

- PARENTING  
- ENTERTAINMENT  
- POLITICS  
- WELLNESS  
- BUSINESS  
- STYLE & BEAUTY  
- FOOD & DRINK  
- QUEER VOICES  
- TRAVEL  
- HEALTHY LIVING

---

## 💻 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/news-category-rnn.git
cd news-category-rnn
