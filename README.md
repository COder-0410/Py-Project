# A Chatbot Powered By TinyLLaMA

This is a simple chatbot built using the **Transformers** library and powered by the **TinyLLaMA-1.1B-Chat-v1.0** model. 

<b><i>Note: It was as part of my academic assignment, to create a web chatbot application using </b>```Flask```<b>. </b></i>

## Features

- TinyLLaMA-powered conversation
- Flask-based web interface
- Efficient CPU/GPU inference using Hugging Face `transformers` and `accelerate`

## Model Info:
 Used: [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
 <br><br>This model is lightweight and ideal for running on resource-constrained environments while still providing strong performance on chat tasks.

## Requirements

 You can view the full list of requirements here:  
 [requirements.txt](./requirements.txt)

Install the required Python packages:

```bash
pip install -r requirements.txt
```
# How to Run:
## 1. Clone the Repository:
```bash
git clone https://github.com/COder-0410/Py-Project.git
cd Py-Project
```
## 2. Run the webui.py:
 On Windows:
 ```pwsh
python webui.py
```
On linux:
```bash
python3 webui.py
```
