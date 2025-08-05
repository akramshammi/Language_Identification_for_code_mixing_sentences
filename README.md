# Language_Identification_for_code_mixing_sentences

This project implements a **code-mixed multilingual language detection system** that can identify the language of a given sentence or even individual words within a sentence.  
It uses **three approaches**:
1. **Multinomial Naive Bayes**
2. **Decision Tree Classifier**
3. **Hidden Markov Model (HMM)** using `hmmlearn`

The model is trained on a dataset containing text samples from multiple languages, enabling detection in scenarios such as **code-mixing** (when multiple languages appear in a single sentence).

---

## 📌 Features
- Detects **the language of a sentence**.
- Detects **the language of individual words** (useful for mixed-language sentences).
- Compares performance of:
  - Naive Bayes Classifier
  - Decision Tree Classifier
  - Hidden Markov Model
- Calculates **Accuracy, Precision, Recall, and F1 Score** for each model.
- Configurable thresholds for HMM to fine-tune predictions.
- Handles unseen tokens gracefully.

---

## 📂 Dataset
- Input file: `Language Detection.csv`
- Contains:
  - `Text`: The sentence or phrase.
  - `Language`: The corresponding language label.
- In this implementation, only the **first 2280 rows** are used for training/testing.

---

## 📦 Installation

Make sure you have Python 3.11+ installed.  
Run the following commands to set up the environment:


pip install pandas scikit-learn nltk langdetect langid hmmlearn
Additionally, download necessary NLTK tokenizers:

python
Copy
Edit
import nltk
nltk.download('punkt')
🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone <your-repo-link>
cd <repo-folder>
Place your dataset

Ensure Language Detection.csv is in the project directory.

Run the script

bash
Copy
Edit
python language_detection.py
Example Interactive Run

css
Copy
Edit
Enter a sentence: hello aap kese ho
hello - English
aap - Malayalam
kese - Malayalam
ho - Malayalam
📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Naive Bayes	0.9912	0.9912	0.9912	0.9911
Decision Tree	0.9649	0.9698	0.9649	0.9651
Hidden Markov Model	0.8925	0.9475	0.8925	0.9136

⚙️ Code Structure
Data Loading
Loads Language Detection.csv and splits into training/testing sets.

Vectorization
Uses CountVectorizer to convert text into numerical features.

Model Training

Trains Naive Bayes and Decision Tree using Scikit-learn.

Trains an individual Gaussian HMM model per language using hmmlearn.

Prediction Functions

predict_languages(text) → Predict sentence language using Naive Bayes + HMM.

predict_sentence_languages(text) → Returns all possible languages in a sentence.

predict_word_languages(text) → Returns predicted language for each word.

Evaluation Metrics
Computes Accuracy, Precision, Recall, and F1 Score for each model.

🧪 Example Outputs
Sentence Detection

less
Copy
Edit
Enter a sentence: malyalam, tamil, kannada, hindi, kokborok, english, punjabi.
Detected Languages: ['Malayalam', 'English', 'Hindi', 'Tamil', 'Kannada', 'Kokborok', 'Punjabi']
Word-by-Word Detection

css
Copy
Edit
Enter a sentence: hello aap kese ho
hello - English
aap - Malayalam
kese - Malayalam
ho - Malayalam
🛠 Dependencies
Python 3.11+

pandas

scikit-learn

nltk

langdetect

langid

hmmlearn

numpy
