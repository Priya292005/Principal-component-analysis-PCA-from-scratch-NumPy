# Principal-component-analysis-PCA-from-scratch-NumPy
Principal component analysis (PCA) from scratch using NumPy
# PCA From Scratch Using NumPy

## Project Overview
This project implements Principal Component Analysis (PCA) entirely from scratch using NumPy. The objective is to understand the mathematical foundations of PCA by manually performing standardization, covariance computation, eigen decomposition, and projection.

## Dataset
A synthetic dataset with 200 samples and 10 numerical features was generated using NumPy. The dataset contains correlated features to demonstrate effective dimensionality reduction.

## Methodology
1. Data standardization
2. Covariance matrix computation
3. Eigenvalue and eigenvector decomposition
4. Sorting principal components by explained variance
5. Projection for K=2 (visualization) and K=3 (validation)
6. Comparison with scikit-learn PCA

## Validation
The explained variance ratios from the scratch implementation were compared against scikit-learn’s PCA. Minor numerical differences were observed due to different internal computation methods (eigen decomposition vs SVD).

## Visualization
A 2D scatter plot was generated using the first two principal components to illustrate variance capture and structure preservation.

## Conclusion
The project demonstrates a correct and complete implementation of PCA from scratch and validates its accuracy against a standard machine learning library.

