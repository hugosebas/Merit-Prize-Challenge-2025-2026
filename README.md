# Merit Prize Challenge 2025-2026

Machine learning notebook for Portuguese fake news detection, developed for the
Machine Learning course at Instituto Superior Tecnico - Taguspark.

The project trains, evaluates, and interprets several binary classification
models on Portuguese news texts. Texts are represented with TF-IDF features, and
the main comparison uses accuracy, precision, recall, and F1-score.

## Repository Contents

| File | Description |
| --- | --- |
| `portuguese_fake_news_detection.ipynb` | Main notebook with data loading, feature extraction, training, evaluation, interpretation, and clustering. |
| `README.md` | Project documentation. |

The dataset files are not included in the repository.

## Expected Data

The notebook expects three CSV files with the following names:

| File | Purpose |
| --- | --- |
| `train.csv` | Training set. |
| `val.csv` | Validation set. |
| `test.csv` | Test set. |

Each CSV must contain at least the following columns:

| Column | Description |
| --- | --- |
| `Text` | News article text. |
| `Label` | Binary news label. In the notebook, `0` corresponds to fake and `1` corresponds to real. |

For local execution, place the three CSV files in the repository root. In Google
Colab, the notebook asks the user to upload the three files.

## Dependencies

The notebook uses Python and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lime
```

The main dependencies are:

- `pandas` and `numpy` for data processing.
- `matplotlib` and `seaborn` for visualization.
- `scikit-learn` for TF-IDF, models, metrics, PCA, K-Means, and permutation
  importance.
- `lime` for local classification explanations.

## How to Run

1. Install the dependencies.
2. Place `train.csv`, `val.csv`, and `test.csv` in the repository root.
3. Open `portuguese_fake_news_detection.ipynb` in Jupyter Notebook, JupyterLab,
   or Google Colab.
4. Run the cells in order.

Local example:

```bash
jupyter notebook portuguese_fake_news_detection.ipynb
```

## Notebook Pipeline

### 1. Training and Evaluation

The notebook starts by loading the training, validation, and test sets. It then
transforms the `Text` column with `TfidfVectorizer`, using:

- `max_features=5000`
- `min_df=10`
- `max_df=0.9`
- `smooth_idf=True`

It then trains and compares the following models:

- Decision Tree
- Gaussian Naive Bayes
- Logistic Regression with L2 regularization
- Logistic Regression with L1 regularization
- Multi-Layer Perceptron (MLP)

Hyperparameter selection is performed with `GridSearchCV`, using F1-score as the
optimization metric.

### 2. Model Interpretation

The notebook includes an interpretation stage with:

- weights from the best Logistic Regression model;
- words most indicative of fake news;
- words most indicative of real news;
- comparison between L1 and L2 regularization;
- local explanations with LIME for Logistic Regression and MLP;
- permutation importance for the MLP.

### 3. Clustering

In the unsupervised stage, the notebook applies K-Means with `k=5` to the TF-IDF
features. The clusters are analyzed through the documents closest to each
centroid and visualized in 2D with PCA.

The semantic labels assigned in the notebook are:

| Cluster | Interpretation |
| --- | --- |
| 0 | War in Ukraine misinformation |
| 1 | General societal and institutional misinformation |
| 2 | COVID-19 health misinformation |
| 3 | Political fake news |
| 4 | Economic policy misinformation |

## Results Recorded in the Notebook

With the data used in the notebook, the test set results were:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.842188 | 0.838831 | 0.846276 | 0.842537 |
| Gaussian Naive Bayes | 0.853890 | 0.846536 | 0.863708 | 0.855036 |
| Logistic Regression L2 | 0.903542 | 0.903840 | 0.902694 | 0.903267 |
| Logistic Regression L1 | 0.900854 | 0.901525 | 0.899525 | 0.900524 |
| Multi-Layer Perceptron (MLP) | 0.918564 | 0.903670 | 0.936609 | 0.919844 |

The best model in the notebook, according to test F1-score, was the Multi-Layer
Perceptron (MLP).

The notebook also generates and saves some figures, including:

- `best_model_roc.png`
- `clusters_by_cluster.png`
- `clusters_by_label.png`

These files are created during execution and are not included in the repository
by default.

## Author

Hugo Sebastiao  
Student no. 79768  
Machine Learning - Taguspark  
TPC group 4  
Merit Prize group 1  
Instituto Superior Tecnico - Taguspark
