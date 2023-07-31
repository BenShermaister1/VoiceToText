import torch.multiprocessing
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from torch import multiprocessing
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

if __name__ == '__main__':
    multiprocessing.freeze_support()

# import the relavant libraries for loggin in
from huggingface_hub import login

def login_hugging_face(token: str) -> None:
    """
    Log in to Hugging Face portal with a given token.
    """
    login(token=token)

    return None

token = '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'

login_hugging_face(token)
print('we are logged in to Huggin Face!')

# STEP 1. Download Dataset
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()



#
# # common_voice["train"] = load_dataset("imvladikon/hebrew_speech_kan", "he", split="train", use_auth_token=True,to=100)
# common_voice["train"] = load_dataset("imvladikon/hebrew_speech_kan", "he", split="train", use_auth_token=True)
# common_voice["test"] = load_dataset("imvladikon/hebrew_speech_kan", "he", split="validation", use_auth_token=True)
#
# print(common_voice)


dataset_config = {
  "LOADING_SCRIPT_FILES": "C:/pythonProject/pythonProjectHF/hebrew_speech_recognition.py",
  "CONFIG_NAME": "he",
  "DATA_DIR": "C:/pythonProject/pythonProjectHF/data/he",
  "CACHE_DIR": "C:/pythonProject/pythonProjectHF/.cache/cache_dataset",
}

common_voice = load_dataset(
  dataset_config["LOADING_SCRIPT_FILES"],
  dataset_config["CONFIG_NAME"],
  data_dir=dataset_config["DATA_DIR"],
  cache_dir=dataset_config["CACHE_DIR"]
)

print(common_voice)

# STEP 2. Prepare: Feature Extractor, Tokenizer and Data
# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("BenShermaister/whisper-base-he")

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("BenShermaister/whisper-base-he", language="Hebrew", task="transcribe")

# STEP 3. Combine elements with WhisperProcessor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("BenShermaister/whisper-base-he", language="Hebrew", task="transcribe")

# STEP 4. Prepare Data

print('| Check the random audio example from Common Voice dataset to see what form the data is in:')
print(f'{common_voice["train"][0]}\n')

# -> (1): Downsample from 48kHZ to 16kHZ
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('| Check the effect of downsampling:')
print(f'{common_voice["train"][0]}\n')

def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# Prepare and use function to prepare our data ready for the Whisper AI model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1 # num_proc > 1 will enable multiprocessing
    )

# STEP 5. Training and evaluation
# STEP 5.1. Initialize the Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5.2. Define evaluation metric
import evaluate
metric = evaluate.load("wer")

# STEP 5.3. Load a pre-trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("BenShermaister/whisper-base-he")

"""
Overide generation arguments:
- no tokens are forced as decoder outputs: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids
- no tokens are suppressed during generation: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens
"""
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# STEP 5.4. Define the training configuration
"""
Check for Seq2SeqTrainingArguments here:
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
"""
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="C:\pythonProject\pythonProjectHF\whisper-base-he",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=2,
    gradient_checkpointing=False,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=0,
    eval_steps=0,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)

# Initialize a trainer.
"""
Forward the training arguments to the Hugging Face trainer along with our model,
dataset, data collator and compute_metrics function.
"""
from transformers import Seq2SeqTrainer

def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


#STEP 5.5. Training
"""
Training will take appr. 5-10 hours depending on your GPU.
"""
print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')

print(trainer)

trainer.save_model('./savemodel')
tokenizer.save_pretrained('./savemodel')

trainer.push_to_hub()

#Step 6.1. connect with write token 
def login_hugging_face(token: str) -> None:
    """
    Log in to Hugging Face portal with a given token.
    """
    login(token=token)

    return None

token = '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'

login_hugging_face(token)
print('we are logged in to Huggin Face!')

#step 6.2. push to hub



trainer.tokenizer.push_to_hub("BenShermaister/whisper-base-he")

trainer.model.push_to_hub("BenShermaister/whisper-base-he")

