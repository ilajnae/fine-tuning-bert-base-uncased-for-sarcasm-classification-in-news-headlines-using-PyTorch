# Fine-Tuning BERT-Base-Uncased for Sarcasm Classification in News Headlines using PyTorch

## Overview
This project focuses on building a sarcasm classification model for news headlines by fine-tuning the pre-trained BERT-Base-Uncased model using PyTorch and Transformers. The goal is to detect sarcasm by leveraging contextual understanding, a critical aspect of natural language processing.

## Dataset
The dataset used is the [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection), which contains labeled news headlines indicating whether each headline is sarcastic or not.

## Methodology
1. **Data Preprocessing:**
   - Downloaded the dataset using the `opendatasets` library.
   - Tokenized text using the BERT tokenizer (`bert-base-uncased`).
2. **Model Setup:**
   - Utilized Hugging Face's Transformers library.
   - Loaded the pre-trained BERT model for sequence classification.
3. **Training:**
   - Fine-tuned BERT-Base-Uncased with PyTorch.
   - Used GPU acceleration (`cuda`) when available.
4. **Evaluation:**
   - Assessed model performance using accuracy and loss plots.


## Why Google Colab? 
Fine-tuning **BERT-Base-Uncased** for sarcasm classification is computationally intensive. The model has **110 million parameters**, requiring significant memory and processing power. Google Colab provides **free GPU acceleration**, making it an ideal choice for training deep learning models.  

### Why is a GPU needed?  
- **Faster Training:** Parallelized computations using CUDA significantly reduce training time.  
- **Memory Handling:** GPUs handle large batch sizes and matrix multiplications more efficiently.  
- **Transformer Optimization:** BERT-based models leverage tensor operations that run much faster on GPUs.  

### Enabling GPU in Colab  
1. **Go to** `Runtime` → `Change runtime type`  
2. **Select** `"GPU"` as the hardware accelerator  
3. **Verify GPU access:**  
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

## Installation
Ensure you have Python 3.8+ installed, then run:

```bash
# Install dependencies (included for you in the notebook as well)
pip install transformers torch opendatasets
```

## Usage
Run the Jupyter notebook: `sarcasm-classifier.ipynb`


## Results
- The model achieved competitive accuracy in sarcasm detection.
- Loss and accuracy plots are visualized for better model understanding.

## Acknowledgments
- Dataset: [Kaggle - News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- Transformers library by Hugging Face
- PyTorch for model fine-tuning
