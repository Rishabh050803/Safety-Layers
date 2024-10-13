# Readme

 This supplemental material provides all the necessary code and datasets required for reproducing the analyses and experiments outlined in our dissertation. The materials are organized into two primary directories: the `Code` folder and the `Dataset` folder. Additionally, a `requirements.txt` file is included to ensure the correct environment setup for running the provided code. 



## Dataset Folder

The `Dataset` folder is structured to contain the datasets used for fine-tuning and validation across various experiments. It includes:

- **Fine-Tuning Data**: This subfolder contains both the backdoor data and the normal dataset, as referenced in Section 4 of our paper.
- **Evaluation Folder**: This folder holds datasets used for model security evaluation:
  1. **Over-Rejection Dataset**: Used in the safety layer localization experiments described in Section 3, this dataset is instrumental in assessing model security when different intervals of layer parameters are scaled.
  2. **Malicious Problem Dataset**: Used to evaluate model security, as discussed in Section 4. 



## Code Folder

The `Code` folder is divided into the following subfolders, each containing scripts to execute specific tasks:

- **Fine-Tuning**: Scripts related to fine-tuning the model, including setting up the training pipeline and handling custom dataset injections.

- **Cosine Similarity Analysis & Plotting**: Code for computing and visualizing cosine similarity metrics, which are crucial for recovering the existence of safety layers.

- **Attention Score Extraction & Plotting**: Contains scripts for extracting each layer's attention scores and visualizing them, aiding in the interpretability analysis. 

  If using a gemma series LLM, uncomment the code on lines 51 and 85 in `att_scores.py`.

- **SPPFT Code**: Specific scripts for implementing and testing the SPPFT technique, as outlined in section 4 of the paper.

- **Safety Layer Localization Pipeline**: Code to reproduce the safety layers localization experiments.

  If using a phi-3 series LLM, uncomment the code in line 133 and comment out the code in line 132.

To run the provided scripts, navigate to the appropriate directory where the code is located, and execute the script with the following command: 

```
python <script_name>.py
```

 By default, the code is set to use the `Llama-2-7b-chat` model. If you wish to use a different model, you can specify the desired model by using the `--model_path` argument, like so: 

```
python <script_name>.py --model_path <path_to_model>
```



 For more details on setting up the environment and ensuring compatibility, please refer to the `requirements.txt` file. 