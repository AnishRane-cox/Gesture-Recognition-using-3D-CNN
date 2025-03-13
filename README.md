# Repository Description: Gesture Recognition using 3D CNN
 This repository contains experiments with various deep learning models for spatiotemporal data classification. The goal is to improve model performance while reducing overfitting and instability.  Key Highlights: Model Variations: Tested different architectures, including Conv3D, ConvLSTM, TimeDistributed Conv2D + LSTM, and hybrid models with GRU. Regularization Techniques: Applied L1/L2 regularization, Batch Normalization, Dropout (spatial and recurrent), and Global Average Pooling to stabilize training. Optimization Strategies: Experimented with learning rate scheduling, data augmentation, and fine-tuning hyperparameters to balance training and validation accuracy. Final Model: Achieved the best validation accuracy (78.00%) by optimizing dropout rates, regularization, and learning rate adjustments. This repository provides code, results, and insights into improving spatiotemporal deep learning models for sequence classification tasks. 

 # Gesture Recognition using 3D CNN  

##  Overview  
This repository contains an implementation of a **Gesture Recognition** model using **3D Convolutional Neural Networks (Conv3D)**. The model is trained on video-based gesture data to classify different hand or body gestures accurately.  

##  Features  
- Uses **Conv3D** layers with **MaxPooling3D** and **Dropout** for feature extraction.  
- Implements **L2 regularization** to reduce overfitting.  
- Experiments with different **dropout rates** and **regularization techniques** for optimization.  
- Achieves a **validation accuracy of 78%**, demonstrating strong generalization.  

##  Experiments & Results  

| Experiment Number | Model Configuration | Regularization | Dropout Rate | Learning Rate | Validation Accuracy | Decision & Explanation |
|------------------|----------------------|----------------|--------------|---------------|----------------------|------------------------|
| 1 | Conv3D (8,16,32) + MaxPooling3D + Dropout | None | 0.25 | 0.001 | 78.00% | Best performing model with optimal dropout and regularization. |
| 2 | Conv3D with L2 Regularization | L2 (0.01) | 0.3 | 0.001 | 68.00% | Regularization added stability but reduced validation accuracy. |
| 3 | Conv3D with Lower Dropout | None | 0.15 | 0.001 | 73.00% | Slight improvement in training accuracy but lower validation performance. |
| 4 | Conv3D (16,32,64) + L2 Regularization | L2 (0.001) | 0.2 | 0.0005 | 75.50% | Improved generalization but slightly lower accuracy. |
| 5 | Final Model (Best Configuration) | None | 0.25 | 0.001 | 78.00% | Selected as the best model due to optimal balance of dropout and regularization. |

### Installation & Usage  

### Clone this repository:  
```bash

git clone https://github.com/AnishRane-cox/Gesture-Recognition-using-3D-CNN
cd Gesture-Recognition
```

### Install dependencies:  
```bash
pip install -r requirements.txt
```

### Run the training script:  
```bash
python train.py
```

### Test the model:  
```bash
python test.py
```

### Results & Conclusion  
The final model achieved a validation accuracy of **78%**, making it a robust approach for real-time gesture recognition. The use of **3D CNNs** effectively captures spatiotemporal features in video sequences, leading to improved performance.  

Feel free to contribute or report issues!

