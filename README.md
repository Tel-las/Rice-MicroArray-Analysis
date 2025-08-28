# Rice Microarray Stress Classification

Authors: Mariana Coelho (PG42876), Mónica Fernandes (PG42878), Tiago Machado (PG42884)

## Overview

This repository contains an analysis pipeline and notebook for classifying biotic vs abiotic stress in Oryza sativa (rice) using microarray gene expression data. The work follows a published meta-analysis (Shaik & Ramakrishna, 2014) to select a panel of differentially expressed genes and applies exploratory statistics, dimensionality reduction, clustering, classical machine learning and deep learning methods.

Key highlights
- Dataset: 219 samples x 1377 genes (DEGs from Shaik & Ramakrishna, 2014)
- Tasks: exploratory analysis, unsupervised clustering, supervised classification, hyperparameter tuning, and deep learning experiments
- Best classical models: K-Nearest Neighbors (KNN) and Random Forest (~97% accuracy)
- Deep learning: DNN/CNN/LSTM tested but showed overfitting due to limited data

## Files in this repository

- `Entrega_M4.ipynb` — main Jupyter notebook with the full analysis and narrative (recommended entry point)
- `Base_data.csv`, `Final_ds.csv`, `DEGs_final.csv`, `Metric_no_FS.csv` — processed data and metric outputs used/produced in the notebook
- `SOFT_F/` — downloaded GEO SOFT files used to build the dataset

## Data sources and citation

The raw expression data were downloaded from the Gene Expression Omnibus (GEO). The feature selection (list of DEGs) is taken from:

Shaik, R. and Ramakrishna, W. (2014). [DOI:10.1104/PP.113.225862]

If you use this repository or derived results, please cite the original work and the GEO series used.

## Methods (high level)

- Data curation: parse GEO SOFT files, average replicates, and filter to the DEGs list (1377 probes)
- Univariate analysis: normality tests, Levene test for variance homogeneity, Z-test and Mann–Whitney U tests
- Dimensionality reduction: PCA, PLS-DA, t-SNE
- Unsupervised learning: K-Means and hierarchical clustering
- Supervised learning: Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest, SVM, MLP, Bagging, AdaBoost
- Hyperparameter optimization: GridSearchCV for Random Forest and KNN
- Deep learning: DNN, LSTM and 1D-CNN architectures (Keras / TensorFlow)

## Key results (summary)

- After preprocessing and feature selection, the dataset has 219 samples and 1377 genes.
- PCA (2 components) explains ~47% of variance; 39 components explain ~95%.
- Random Forest and KNN reached ~96.97% test accuracy after tuning.
- Deep learning models achieved around ~89–90% accuracy but showed overfitting given the small dataset.

## Reproduce the analysis (quickstart)

1. Clone the repository and change to its folder.
2. Install dependencies (suggested virtual environment recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Open the notebook `Entrega_M4.ipynb` in Jupyter Lab or Notebook and run the cells in order. The notebook includes data loading, preprocessing, analysis and plotting.

Notes:
- The original SOFT files (GEO series) are present under `SOFT_F/` if you want to re-run the dataset generation steps. Some cells that fetch data from GEO may be disabled or commented out; inspect the notebook before re-running heavy downloads.
- Depending on your machine and whether you retrain models (GridSearch, DL training), runtime and memory requirements may be high.

## Dependencies

Install the required Python packages with the `requirements.txt` provided. Primary libraries used include: pandas, numpy, SciPy, scikit-learn, statsmodels, GEOparse, seaborn, matplotlib, tensorflow/keras.

## Reproducing ML experiments (notes)

- The notebook splits data using train_test_split with a fixed random seed for reproducibility.
- Cross-validation uses RepeatedKFold (10 folds x 3 repeats).
- GridSearchCV was used for selected algorithms; results are written to CSV files in the repository when executed.

## Recommendations & caveats

- The dataset is relatively small for deep learning; DL models show overfitting and classical ML methods (KNN, Random Forest) performed better here.
- If you plan to extend this work: add more transcriptomic datasets, perform nested CV for robust hyperparameter selection, and add model persistence (joblib) and unit tests for data pipelines.

## License & Contact

This repository is provided for academic purposes. Contact the authors for collaboration or questions (see notebook header for names).
