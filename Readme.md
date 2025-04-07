# Safety Layers in Aligned Large Language Models: The Key to LLM Security 

 This repository is the official implementation of the paper  [Safety Layers in Aligned Large Language Models: The Key to LLM Security](https://arxiv.org/abs/2408.17003) (Accepted by ICLR 2025)  

The materials are organized into two primary directories: the `Code` folder and the `Dataset` folder. The "requirement.txt" file is in the code folder.

## Requirements

To get started, please clone this repository and install packages as:

```python
git clone https://github.com/listen0425/Safety-Layers.git
conda create -n Safety_layers python=3.10
...
pip install -r requirements.txt
```

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



## Citation

If you find this code useful, please kindly cite our work as:

```
@article{li2024safety,
  title={Safety layers in aligned large language models: The key to llm security},
  author={Li, Shen and Yao, Liuyi and Zhang, Lan and Li, Yaliang},
  journal={arXiv preprint arXiv:2408.17003},
  year={2024}
}
```
