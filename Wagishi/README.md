# Gray Interface 25 – Wagishi

This folder contains my submissions and learning projects for **Gray Interface 25** under **HackSlash NIT Patna**.

The projects cover Python fundamentals, data analysis, machine learning, natural language processing, neural networks, and deep learning.

---

## About

This repository is a collection of task-based projects completed as part of my learning journey in **AI/ML and Python programming**.

Each task focuses on a different concept, starting from basic vector operations and progressing toward machine learning models, NLP, neural networks, and neural style transfer.

---

## Project Index

| Task          | Project                                     | File                                |
| ------------- | ------------------------------------------- | ----------------------------------- |
| Task 1        | Vector Distance and Angle Computation       | `Task1.py`                          |
| Task 2.1      | Diabetes Prediction Model                   | `Tak2.1_diabetes.py`                |
| Task 2.2      | Spotify Streams Prediction                  | `Task2_2_Spotify.ipynb`             |
| Task 3        | Movie Review Sentiment Analysis             | `Movie_review_Model_Creation.ipynb` |
| Task 4 Week 1 | Neural Network Forward Pass for Addition    | `Task4(Week_1).ipynb`               |
| Task 4 Week 2 | Neural Network Backpropagation for Addition | `task_4(week_2).py`                 |
| Task 5        | Neural Style Transfer using TensorFlow      | `hackslashtask_5.py`                |

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* NLTK
* WordCloud
* TensorFlow
* Keras
* VGG19
* Google Colab
* Jupyter Notebook

---

# Task 1 – Vector Distance and Angle Computation

**File:** `Task1.py`

## Overview

This task implements basic mathematical operations used in machine learning and data science for comparing vectors.

The program calculates:

* Euclidean Distance
* Manhattan Distance
* Cosine Distance
* Angle between two vectors

## Concepts Used

* NumPy arrays
* Vector operations
* Dot product
* Vector norm
* Cosine similarity
* Angle calculation using inverse cosine

## Learning Outcome

This task helped me understand how distance metrics are used to compare data points in machine learning algorithms such as KNN, clustering, and recommendation systems.

---

# Task 2.1 – Diabetes Prediction Model

**File:** `Tak2.1_diabetes.py`

## Overview

This project uses machine learning to predict diabetes risk based on health-related features from the diabetes dataset.

## Features Used

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age
* Outcome

## Workflow

1. Loaded the dataset using Pandas.
2. Explored the dataset using `.head()`, `.info()`, `.shape()`, and `.describe()`.
3. Checked missing values.
4. Visualized feature distributions using histograms and KDE plots.
5. Analyzed feature correlation using a heatmap.
6. Split the data into training and testing sets.
7. Standardized features using `StandardScaler`.
8. Trained machine learning models.
9. Evaluated model performance.

## Models Used

* Logistic Regression
* Linear Regression

## Learning Outcome

This project helped me understand the basic machine learning pipeline, including data exploration, visualization, preprocessing, model training, and evaluation.

---

# Task 2.2 – Spotify Streams Prediction

**File:** `Task2_2_Spotify.ipynb`

## Overview

This project predicts the number of streams a song may receive on Spotify based on music platform statistics and song-related features.

## Important Features

* Artist name
* Track name
* Release year
* Spotify playlists
* Spotify charts
* Apple playlists
* Apple charts
* Deezer playlists
* Deezer charts
* Shazam charts
* BPM
* Danceability
* Energy
* Acousticness
* Liveness
* Speechiness

## Workflow

1. Loaded the Spotify dataset.
2. Explored the data using Pandas.
3. Checked missing values.
4. Analyzed correlations between features.
5. Selected useful numerical features.
6. Trained a regression model.
7. Predicted song streams.
8. Created a predictions DataFrame containing artist name, track name, and predicted streams.

## Model Used

* Linear Regression

## Learning Outcome

This project helped me understand regression problems, feature correlation, dataset cleaning, and how machine learning can be used for prediction tasks in the music industry.

---

# Task 3 – Movie Review Sentiment Analysis

**File:** `Movie_review_Model_Creation.ipynb`

## Overview

This project performs sentiment analysis on movie reviews. The goal is to classify movie reviews as either positive or negative using Natural Language Processing techniques.

## Features

* `text` – Movie review content
* `sentiment` – Target label

## Workflow

1. Loaded the movie review dataset.
2. Cleaned and preprocessed the text data.
3. Removed stop words.
4. Visualized important words using WordCloud.
5. Converted text into numerical features using TF-IDF.
6. Trained a Logistic Regression model.
7. Evaluated the model using a confusion matrix.
8. Saved the trained model and vectorizer using Pickle.

## Libraries Used

* Pandas
* Matplotlib
* NLTK
* WordCloud
* Scikit-learn
* Pickle

## Concepts Used

* Natural Language Processing
* Text preprocessing
* Stop word removal
* TF-IDF vectorization
* Logistic Regression
* Model evaluation
* Model saving

## Learning Outcome

This project helped me understand how machine learning can be applied to text data and how sentiment analysis works in real-world applications such as review classification, social media analysis, and recommendation systems.

---

# Task 4 – Neural Network for Addition

## Overview

This project implements a simple neural network from scratch using NumPy. The goal is to train a model that learns the addition of two numbers.

This task is divided into two parts:

* Week 1: Forward Pass
* Week 2: Backpropagation

---

## Task 4 Week 1 – Forward Pass

**File:** `Task4(Week_1).ipynb`

## Overview

In Week 1, the neural network performs only forward propagation.

## Network Structure

* Input Layer: 2 input values
* Hidden Layer: 2 neurons
* Output Layer: 1 neuron

## Concepts Used

* Random weight initialization
* Bias initialization
* Sigmoid activation function
* Forward propagation
* Prediction generation

## Learning Outcome

This task helped me understand how data flows through a neural network from the input layer to the hidden layer and then to the output layer.

---

## Task 4 Week 2 – Backpropagation

**File:** `task_4(week_2).py`

## Overview

In Week 2, backpropagation is implemented to train the neural network.

## Network Structure

* Input Layer: 2 input values
* Hidden Layer: 1 neuron
* Output Layer: 1 neuron

## Concepts Used

* Forward pass
* Loss calculation
* Sigmoid derivative
* Gradient calculation
* Backpropagation
* Weight and bias updates
* Training loop

## Example Training Data

```python
X = np.array([
    [2, 2],
    [3, 4],
    [5, 7],
    [10, 15]
])

y = np.array([
    [4],
    [7],
    [12],
    [25]
])
```

## Learning Outcome

This task helped me understand how neural networks learn by reducing prediction error through backpropagation and gradient-based weight updates.

---

# Task 5 – Neural Style Transfer using TensorFlow

**File:** `hackslashtask_5.py`

## Overview

This project implements Neural Style Transfer using TensorFlow and a pretrained VGG19 convolutional neural network.

Neural Style Transfer combines the content of one image with the artistic style of another image to generate a new stylized image.

## Concepts Used

* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* VGG19
* Feature extraction
* Content loss
* Style loss
* Gram matrix
* GradientTape
* Image optimization

## Workflow

1. Loaded content and style images.
2. Preprocessed images for VGG19.
3. Used pretrained VGG19 without the top classification layer.
4. Extracted content features from deeper layers.
5. Extracted style features from multiple convolutional layers.
6. Calculated style representation using Gram Matrix.
7. Computed total loss using content loss and style loss.
8. Used gradient descent to optimize the generated image.
9. Displayed the final stylized image.

## Features

* Uses pretrained VGG19 model
* Supports custom content and style images
* Uses content and style loss functions
* Generates a new stylized image
* Visualizes feature maps from CNN layers

## Learning Outcome

This project helped me understand how deep learning models can be used creatively for image generation and artistic transformation.

---

# How to Run the Projects

## 1. Clone the repository

```bash
git clone https://github.com/HackSlashNITP/Gray-Interface-25.git
```

## 2. Go to the Wagishi folder

```bash
cd Gray-Interface-25/Wagishi
```

## 3. Install required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud tensorflow
```

## 4. Run Python files

```bash
python Task1.py
python Tak2.1_diabetes.py
python task_4\(week_2\).py
python hackslashtask_5.py
```

## 5. Run notebook files

Open the `.ipynb` files in Jupyter Notebook or Google Colab:

* `Task2_2_Spotify.ipynb`
* `Movie_review_Model_Creation.ipynb`
* `Task4(Week_1).ipynb`

---

# Folder Structure

```text
Wagishi/
│
├── README.md
├── Task1.py
├── Tak2.1_diabetes.py
├── Task2_2_Spotify.ipynb
├── Movie_review_Model_Creation.ipynb
├── Task4(Week_1).ipynb
├── task_4(week_2).py
├── hackslashtask_5.py
└── hello.txt
```

---

# Key Learnings

Through these tasks, I learned:

* How to perform vector-based mathematical operations using NumPy
* How to explore and visualize datasets
* How to build basic machine learning models
* How regression models can be used for prediction
* How NLP is used for sentiment analysis
* How neural networks perform forward propagation
* How backpropagation updates weights
* How CNN-based models like VGG19 can be used for Neural Style Transfer

---

# Future Improvements

Some possible improvements for these projects are:

* Add proper dataset links in each project section.
* Add accuracy, R² score, and confusion matrix screenshots.
* Improve preprocessing for better model performance.
* Add comments and docstrings in all Python files.
* Add separate README files for large projects.
* Add `requirements.txt` for easier setup.
* Organize datasets inside a `data/` folder.
* Add output images for visual tasks like Neural Style Transfer.

---

# Author

**Wagishi Jagat**



