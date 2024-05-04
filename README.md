# Speech Understanding Programming Assignment 3

**Name:** Kushal Agrawal  
**Roll No:** M23CSA011

- **Gradio Link:** [Audio-deep-fake](https://huggingface.co/spaces/kushal1506/Audio-Deep-fake)
- **Github Link:** [Github](https://github.com/M23CSA011/Audio-Deep-Fake/tree/main)

## Task 1 and 2
Use the SSL W2V model trained for LA and DF tracks of the ASVSpoof dataset. Report the AUC and EER on this dataset.

- Download the XLS-R 300 M model [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) and the Pre-trained SSL anti-spoofing models [here](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB)
- Download the Custom dataset [here](https://iitjacin-my.sharepoint.com/personal/ranjan_4_iitj_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Franjan%5F4%5Fiitj%5Fac%5Fin%2FDocuments%2FDataset%5FSpeech%5FAssignment%2Ezip&parent=%2Fpersonal%2Franjan%5F4%5Fiitj%5Fac%5Fin%2FDocuments&ga=1)
- Pip install the requirements.txt file
- Log in Wandb from terminal using `wandb login “api key”` command

**Execution Instructions:**
1. Unzip the custom dataset and save it in the `SSL_Anti-spoofing` folder.
2. Run the task 1&2 by entering `python evaluate_custom.py` command in the terminal.

**Results:**
- AUC: 0.611
- EER: 0.357

## Code Overview
### Importing Libraries:
- Imports necessary libraries such as os, torch, librosa, numpy, tqdm, sklearn.metrics, and torch.nn.

### Utility Functions:
- Defines two utility functions: `pad(x, max_len)` and `preprocess_audio(audio_path)`.

### Data Preparation:
- Defines paths to directories containing real and fake audio files.
- Creates lists of file paths for real and fake audio samples.

### Model Initialization:
- Determines the device ('cuda' if available, else 'cpu') for model execution.
- Initializes an instance of the model and moves it to the specified device.
- Loads the model's state dictionary from a saved checkpoint file.

### Model Evaluation:
- Sets the model to evaluation mode.
- Processes each audio file and predicts the probability of it being fake.
- Calculates evaluation metrics (AUC, EER) using predictions and ground truth labels.

## Task 3
**Analyze the performance of the model.**

### AUC (Area Under the Curve):
- AUC of 0.611 indicates the model performs better than random chance but has room for improvement.
- Falls below a strong classifier's performance suggesting moderate reliability.

### EER (Equal Error Rate):
- EER of 0.357 indicates relatively balanced performance between false acceptance and false rejection.

### Possible Actions:
- Fine-tune model architecture, optimize hyperparameters, or explore different training strategies.
- Increase diversity and size of the training dataset.

## Task 4 and 5
**Finetune the model on FOR dataset. Report the performance using AUC and EER on FOR dataset.**

- Download the FOR dataset [here](https://www.eecs.yorku.ca/~bil/Datasets/for-2sec.tar.gz)
- Unzip the file in the `SSL_Anti-spoofing` folder.
- Run the task 4&5 by entering `python fine_tune_for.py` command in the terminal.

**Fine-tuning Details:**
- 5 epochs
- Learning rate = 5e-5

**Results:**
- AUC: 0.9738
- EER: 0.084

## Task 6
**Use the model trained on the FOR dataset to evaluate the custom dataset. Report the EER and AUC**

- Run `python evaluate_custom_for.py` command in terminal.

**Results:**
- AUC: 0.113
- EER: 0.833

## Task 7
**Comment on the change in performance**

The significant decrease in AUC from 0.611 to 0.113 and increase in EER from 0.357 to 0.833 after fine-tuning on the custom dataset indicates a substantial deterioration in the model's performance. This change suggests that the fine-tuning process may not have been effective or may have introduced biases or overfitting to the custom dataset. Possible reasons for this decline in performance could include insufficient data for fine-tuning, a mismatch between the pre-trained model and the custom dataset, or inadequate hyperparameter tuning during fine-tuning.
