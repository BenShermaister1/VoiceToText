from datasets import list_datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import huggingface_hub
import tarfile
import os



######################### PRAPRING_BEFOR_UPLOAD_DATASET ############################
dataset_config = {
  "LOADING_SCRIPT_FILES": "C:/pythonProject/pythonProjectHF/hebrew_speech_recognition.py",
  "CONFIG_NAME": "he",
  "DATA_DIR": "C:/pythonProject/pythonProjectHF/data/he",
  "CACHE_DIR": "C:/pythonProject/pythonProjectHF/.cache/cache_dataset",
}

ds = load_dataset(
  dataset_config["LOADING_SCRIPT_FILES"],
  dataset_config["CONFIG_NAME"],
  data_dir=dataset_config["DATA_DIR"],
  cache_dir=dataset_config["CACHE_DIR"]
)

print(ds)