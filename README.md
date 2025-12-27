# Handwriting Recognition using CRNN

**Members:**
Hazem Sobhy Elsayed
Mustafa Mahmoud Elgendy 
Mohamed Abdrabo Ebaid
Omar Ahemd Elabd
Ibrahem Hesham Ibrahem

**Date:** 2025-12-27  

This project implements a Handwriting Recognition system using a Convolutional Recurrent Neural Network (CRNN) and provides a Streamlit GUI for predictions.

---

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)  
2. [Model Construction](#model-construction)  
3. [Model Training](#model-training)  
4. [Evaluation and Test](#evaluation-and-test)  
5. [GUI for Prediction](#gui-for-prediction)  

---

## Data Preprocessing

1. Load CSV files for training, validation, and testing with columns `FILENAME` and `IDENTITY`.  
2. Remove rows with missing data.  
3. Resize images to fixed dimensions (default: 64x256).  
4. Apply transformations:  
   - Grayscale conversion  
   - Random rotation and affine transforms (**train only**)  
   - Gaussian blur, color jitter (**train only**)  
   - Convert to tensor and normalize  
5. Encode labels using character set indices.  
6. Prepare `DataLoader` with padding for CTC loss.  

---

## Model Construction

1. Build CRNN model:  
   - CNN backbone: 4 convolutional layers with BatchNorm, ReLU, MaxPool  
   - RNN: 2-layer bidirectional LSTM, hidden size 256  
   - Fully connected layer mapping to number of classes  
2. Output is log-softmax for CTC loss.  
3. Define CTC blank index for loss computation.  
4. Implement greedy and beam search decoding methods.  

---

## Model Training

1. Define Adam optimizer with learning rate (default: 0.0001).  
2. Train model for specified epochs.  
3. For each batch:  
   - Forward pass  
   - Compute CTC loss  
   - Backpropagation and optimizer step  
4. Track average loss and optionally Character Error Rate (CER).  
5. Save checkpoints after each epoch for resuming training.  

---

## Evaluation and Test

1. Evaluate model on validation or test set using CTC loss.  
2. Print sample predictions vs ground truth.  
3. Use greedy or beam search for decoding.  
4. Calculate average loss and optionally accuracy or CER.  

---

## GUI for Prediction

1. Streamlit GUI allows users to upload handwriting images.  
2. Load trained model using caching to avoid repeated loading.  
3. Transform uploaded images (resize, grayscale, normalize).  
4. Perform prediction using beam search decoding.  
5. Display predicted text and confidence score.  
6. Streamlit components used: `st.title`, `st.file_uploader`, `st.image`, `st.button`, `st.write`.  

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit GUI
streamlit run app.py
