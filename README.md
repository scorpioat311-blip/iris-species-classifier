# **🌸 Iris Flower Classifier **

Teaching a machine to tell flowers apart — no manual rules, just data.
A small supervised learning project that classifies Iris flowers into three species (Setosa, Versicolor, Virginica) using K-Nearest Neighbors. Built as part of an AI internship to get hands-on with the full ML pipeline: scaling, splitting, training, and evaluating — before moving on to anything fancier like neural networks.

🚀 Features

Loads and inspects the classic Iris dataset (150 samples, 4 features, 3 classes)
Scales features with StandardScaler so no single measurement dominates
Splits data into train/test sets (80/20) with shuffling to avoid order bias
Trains a KNN classifier (k=5) and generates predictions
Evaluates results with a confusion matrix and F1 score — not just raw accuracy

🛠️ Tech Stack

Python 3.10
scikit-learn
pandas
matplotlib

⚡ Quick Start

1.Clone the repo and install dependencies:
```bash
git clone https://github.com/scorpioat311-blip/iris-species-classifier.git
```
2.Install dependencies 
```bash
cd iris-knn-classifier
pip install -r requirements.txt
```
3.Run the script:
```bash
Iris_Flower_Classifier.py
```

📂 Project Structure

iris-knn-classifier/
├── knn_project.py       # main script: load, scale, split, train, evaluate
├── requirements.txt     # dependencies
└── README.md

📈 Results

On the test split, the model classifies all three species with strong precision and recall — the confusion matrix comes out nearly clean, since Iris is a well-separated, low-noise dataset. Real-world data won't be this forgiving, which is exactly why the evaluation step (not just accuracy) matters.

🎓 What I Learned

This was my first time building a full ML pipeline from scratch instead of just running a demo notebook. A few things that stuck with me:
Why feature scaling isn't optional — without it, one column can silently dominate the whole model
How picking the right k in KNN is a balancing act between overfitting and underfitting
Why "99% accuracy" can be a trap, especially on imbalanced data — confusion matrix and F1 score tell the real story
Next up: extending this to a more complex dataset and eventually moving into computer vision.

⭐ If you found this useful, consider giving it a star!
