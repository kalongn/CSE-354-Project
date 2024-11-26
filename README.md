# Author Sentiment Analysis task
original paper: https://arxiv.org/pdf/2011.06128   
dataset: https://huggingface.co/datasets/community-datasets/per_sent   
Original source for codebase: CSE 354 Fall 2024 HW3 and inspiration from [the code from the original paper](https://github.com/StonyBrookNLP/PerSenT/blob/main/MyBert_paragraph_document_TPU.ipynb) which was unfortunately outdated hence we used HW3 code to compensate for that.

Contributors: Ka Long Ngai, Jaglin Parmar, TingYue Dong

## Software requirement
This ipynb is expected to be running on Google Colab with T4 GPU at the time of making (Fall 2024 Semester)
#### Software Requirements:
1. Python (Colab default is compatible; generally version >=3.6)
1. Google Colab Environment (pre-installed libraries for Python)

#### Required Python Packages:
1. Transformers: Version: 4.37.0 (Explicitly specified in the code)
2. Datasets
3. Torch (PyTorch): Comes pre-installed in Colab, but ensures compatibility.
4. Pandas: Comes pre-installed in Colab.
5. Scikit-learn: Comes pre-installed in Colab.
6. TQDM: Comes pre-installed in Colab.
7. Numpy: Comes pre-installed in Colab.
8. OS: Built-in Python library (no installation required).

## Models
Here's the link to the [google drive](https://drive.google.com/drive/folders/18u13Ix8CjAzmZAZ8DlvexqzX0MMkrgVV?usp=drive_link) containing all the models

### Baseline model training (distilbert_per_sent_baseline)
1. Using pre-trained [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) model
2. Simply feed labeled (document, true sentiment) training data pair into the model from the training_data from the dataset to train the model.
3. Validate each time with the validation_data and save the best model to Google Drive.
4. Testing on testing_random_data and testing_fixed_data.

