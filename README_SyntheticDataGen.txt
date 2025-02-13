# Synthetic Data Generator for Machine Learning & Optimization

This repository provides MATLAB code to  generate synthetic datasets  for machine learning and optimization applications. The generated data is  Gaussian-distributed , processed using  Principal Component Analysis (PCA) , and evaluated using various  benchmark functions  such as the  Rosenbrock, Mishra, Townsend, Gomez, Simionescu, and Booth functions .

---

## Features
-  Synthetic Data Generation : Generates Gaussian-distributed datasets for two classes.
-  Principal Component Analysis (PCA) : Extracts key features from the generated data.
-  Mathematical Function Evaluations : Computes values from common optimization test functions.
-  Data Partitioning : Splits datasets into  Training (50%) ,  Adaptation (30%) , and  Testing (20%)  sets.
-  Automatic Excel Export : Saves structured data into `.xlsx` files for further analysis.

---

## How It Works
###  1️: Generate Synthetic Data 
-  The script creates two synthetic  Gaussian-distributed  data clusters with predefined means and standard deviations.
-  Rotation Matrices  are applied to align data along principal axes.
-  PCA via Singular Value Decomposition (SVD)  is used for feature extraction.

###  2️: Evaluate Test Functions 
Each data point is evaluated using the following  benchmark functions :
-  Rosenbrock Function 
-  Mishra’s Bird Function 
-  Townsend Function
-  Gomez and Levy Function
-  Simionescu Function
-  Booth Function

###  3️: Data Partitioning 
-  Training Set (50%)  – Used for model training.
-  Adaptation Set (30%)  – Used for parameter fine-tuning.
-  Test Set (20%)  – Evaluates the final model.

###  4️: Export to Excel 
The final datasets are saved into `.xlsx` files:
