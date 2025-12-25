# ECG Arrhythmia Classification Using STFT-Based Spectrogram and Convolutional Neural Network

## Project Overview

This project implements the paper **"ECG Arrhythmia Classification Using STFT-Based Spectrogram and Convolutional Neural Network"** as part of the Biosignal Processing (BSP) course. The goal is to classify ECG signals into five different cardiac arrhythmia categories using advanced signal processing and deep learning techniques.

### Project Classification Classes
- **NOR**: Normal beats
- **LBB**: Left Bundle Branch Block
- **RBB**: Right Bundle Branch Block
- **PVC**: Premature Ventricular Contraction
- **APC**: Atrial Premature Contraction

## Objectives & Requirements

Based on the project guidelines, this implementation covers:

### ✅ Core Deliverables
1. **Understanding of Biosignal Processing Core Concepts**
   - ECG signal acquisition and preprocessing
   - Time-frequency analysis using STFT (Short-Time Fourier Transform)
   - Spectrogram-based feature extraction
   - Deep learning for cardiac signal classification

2. **Implementation Details**
   - MIT-BIH Arrhythmia Database for dataset
   - Signal resampling and windowing (10-second segments)
   - STFT-based spectrogram generation
   - CNN architecture for classification
   - Model training and evaluation

3. **Paper Reproduction & Improvements**
   - Replication of baseline results from the original paper
   - Additional enhancements and novel approaches
   - Performance comparison with paper results

## Dataset

- **Source**: MIT-BIH Arrhythmia Database
- **Sampling Rate**: 360 Hz
- **Window Size**: 10 seconds (3600 samples)
- **Train/Test Split**: As specified in paper (Table 1)
  - Training samples: 2,040 total across 5 classes
  - Test samples: 410 total across 5 classes

### Records Used Per Class
```
NOR: [100, 105, 215]
LBB: [109, 111, 214]
RBB: [118, 124, 212]
PVC: [106, 233]
APC: [207, 209, 232]
```

## Methodology

### 1. Data Preprocessing
- Read ECG signals from MIT-BIH database using WFDB library
- Resample to standardized 360 Hz sampling rate
- Extract 10-second windows centered on annotated beat positions
- Normalize and standardize signal values

### 2. Signal Processing (STFT)
- Apply Short-Time Fourier Transform to ECG segments
- Generate time-frequency spectrograms
- Use appropriate window functions (Hann/Hamming window)
- Compute magnitude spectrograms for feature representation

### 3. Feature Extraction
- Spectrogram normalization (log scale)
- Dimension reduction if needed
- Standardization across dataset

### 4. Model Architecture
- **Base CNN**: Convolutional layers with ReLU activation
- **Pooling**: Max pooling for spatial reduction
- **Dense Layers**: Fully connected layers for classification
- **Regularization**: Dropout and batch normalization
- **Output**: Softmax activation for 5-class classification

### 5. Training & Evaluation
- Optimizer: Adam with appropriate learning rate
- Loss Function: Categorical Cross-Entropy
- Metrics: Accuracy, Precision, Recall, F1-score
- Validation: Cross-validation and separate test set evaluation

## Project Structure

```
.
├── README.md                                          # This file
├── BSP_paper.ipynb                                   # Main implementation notebook
├── Paper_results_replication_and_the_improvements.ipynb  # Results and enhancements
├── FiInalPresentation.pptx                           # Final presentation (10-15 min)
├── Report.pdf                                        # Final report (<15 pages)
├── ECG_Arrhythmia_Classification_Using_STFT-Based_Spectrogram_and_Convolutional_Neural_Network.pdf  # Original paper
├── Project Guidelines.pdf                            # Course project requirements
```

## Notebooks Overview

### 1. **BSP_paper.ipynb** - Core Implementation
Contains the fundamental implementation of the paper methodology:
- Dataset loading and preprocessing
- STFT computation
- CNN model definition and training
- Baseline results generation
- Performance metrics calculation

### 2. **Paper_results_replication_and_the_improvements.ipynb** - Results & Enhancements
Extends the baseline with:
- Results reproduction and comparison
- Novel improvements and modifications
- Additional evaluation metrics
- Visualization of results
- Performance analysis and discussion

## Key Features

✅ **Reproducibility**
- Fixed random seeds (SEED = 42)
- Detailed hyperparameter documentation
- Standardized data splitting

✅ **Comprehensive Evaluation**
- Confusion matrices
- Per-class performance metrics
- ROC curves and AUC scores
- Training/validation loss curves

✅ **Documentation**
- Inline code comments
- Function docstrings
- Clear variable naming
- Output annotations

## Requirements

```
numpy
tensorflow >= 2.0
wfdb
matplotlib
scikit-learn
pandas
scipy
```

## Installation & Usage

### 1. Install Dependencies
```bash
pip install numpy tensorflow wfdb matplotlib scikit-learn pandas scipy
```

### 2. Run the Notebooks
```bash
# Option 1: Jupyter Notebook
jupyter notebook BSP_paper.ipynb

# Option 2: Jupyter Lab
jupyter lab BSP_paper.ipynb

# Option 3: VS Code with Jupyter extension
# Open notebooks in VS Code and run cells
```

### 3. Expected Output
- Training progress logs
- Validation metrics per epoch
- Test set performance
- Visualization plots (spectrograms, confusion matrices, ROC curves)
- Performance comparison tables

## Results

### Baseline Performance
Results are compared against the original paper's metrics:
- Overall Classification Accuracy
- Per-class Sensitivity and Specificity
- Confusion Matrix Analysis

### Improvements & Novelty
Additional experiments include:
- [Details of improvements will be added based on your notebook content]
- Enhanced preprocessing techniques
- Model architecture variations
- Data augmentation strategies

## Challenges & Solutions

### Challenge 1: Data Imbalance
- Classes have different sample counts (PVC: 360, others: 540)
- Solution: Class weighting and balanced sampling

### Challenge 2: Temporal Features
- ECG signals contain temporal dependencies
- Solution: STFT captures frequency evolution over time

### Challenge 3: Overfitting
- Limited dataset size with many parameters
- Solution: Dropout, batch normalization, early stopping, data augmentation

## Learning Outcomes

Through this project, we gained understanding of:
1. **Biosignal Processing**: ECG signal characteristics and arrhythmia patterns
2. **Time-Frequency Analysis**: STFT fundamentals and spectrogram interpretation
3. **Deep Learning**: CNN architecture design for signal classification
4. **Medical AI**: Challenges in clinical signal classification
5. **Model Evaluation**: Comprehensive metrics beyond simple accuracy

## Strengths of the Method

- **Effective Feature Representation**: STFT spectrograms capture both temporal and spectral information
- **Automated Learning**: CNN automatically learns relevant features without manual engineering
- **Interpretability**: Spectrograms provide visual insight into classification features
- **Generalization**: Transferable approach to other biosignal classification tasks

## Limitations & Future Improvements

### Current Limitations
- Relatively small training dataset
- Single channel ECG analysis
- Fixed window size approach
- Limited architectural variations

### Possible Improvements
- Multi-lead ECG analysis (use all available channels)
- Adaptive window sizing based on heartbeat
- Advanced architectures (ResNets, Attention mechanisms)
- Transfer learning from other ECG datasets
- Ensemble methods combining multiple models
- Explainability analysis (saliency maps, LIME)
- Real-time inference optimization

## References

1. **Original Paper**: "ECG Arrhythmia Classification Using STFT-Based Spectrogram and Convolutional Neural Network"
2. **MIT-BIH Database**: Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
3. **WFDB**: Physionet WFDB Documentation
4. **TensorFlow/Keras**: Deep Learning Framework Documentation

## Authors

-Rebecca Fernando
-Weijith Wimalasiri

## Course Information

- **Course**: Biosignal Processing (BSP)
- **Semester**: 7
- **University**: University of Moratuwa
- **Year**: 2025

## License

This project is for educational purposes as part of the BSP course.

---


**Last Updated**: December 2024  
**Status**: Implementation Complete | Results Documented | Ready for Presentation
