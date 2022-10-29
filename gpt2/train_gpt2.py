# TODO
# - create the dataset
# - add caching and multi-processing to the dataset.map() calls
# - train a basic model on the dataset
# - evaluate the validation loss of the trained model
# - train a sequence of models and produce a scaling plot
#
 
import os
import sys
import itertools
import torch
import datasets
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

#import c4

num_processing_workers = 8
data_root_dir = "/mnt/disk2"

#dataset_name = "wikitext"
dataset_name = "c4"

# From the GPT3 paper, Table 2.1
GPT3_CONFIGS = {
    "small": {
        "n_params": 125e6,
        "n_layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "d_head": 64,
        "batch_size": 0.5e6,
        "learning_rate": 6.0e-4,
    },
    "medium": {
        "n_params": 350e6,
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "d_head": 64,
        "batch_size": 0.5e6,
        "learning_rate": 3.0e-4,
    },
    "large": {
        "n_params": 760e6,
        "n_layers": 24,
        "d_model": 1536,
        "n_heads": 16,
        "d_head": 96,
        "batch_size": 0.5e6,
        "learning_rate": 2.5e-4,
    },
    "xl": {
        "n_params": 1.3e9,
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 24,
        "d_head": 128,
        "batch_size": 1e6,
        "learning_rate": 2.0e-4,
    },
    "2.7b": {
        "n_params": 2.7e9,
        "n_layers": 32,
        "d_model": 2560,
        "n_heads": 32,
        "d_head": 80,
        "batch_size": 1e6,
        "learning_rate": 1.6e-4,
    },
    "6.7b": {
        "n_params": 6.7e9,
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
        "d_head": 128,
        "batch_size": 2e6,
        "learning_rate": 1.2e-4,
    },
    "13b": {
        "n_params": 13e9,
        "n_layers": 40,
        "d_model": 5140,
        "n_heads": 40,
        "d_head": 128,
        "batch_size": 2e6,
        "learning_rate": 1.0e-4,
    },
    "175b": {
        "n_params": 175e9,
        "n_layers": 96,
        "d_model": 12288,
        "n_heads": 96,
        "d_head": 128,
        "batch_size": 3.2e6,
        "learning_rate": 0.6e-4,
    },
}

def get_gpt2_config(gpt3_name):
    config = GPT3_CONFIGS[gpt3_name]
    return GPT2Config(
        n_embd=config["d_model"],
        n_layer=config["n_layers"],
        n_head=config["n_heads"],
    )

print("Creating tokenizer and model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# TODO
from_pretrained = False

if from_pretrained:
	model = GPT2LMHeadModel.from_pretrained("gpt2")
	model_config = GPT2Config()
	#print(model_config)
else:
	model_config = get_gpt2_config("small")
	model = GPT2LMHeadModel(model_config)

context_len = model_config.n_positions
num_parameters = model.num_parameters(only_trainable=True, exclude_embeddings=True)
chinchilla_optimal_tokens = 20 * num_parameters

print("context_len =", context_len)
print("num_parameters =", num_parameters)
print("chinchilla_optimal_tokens =", chinchilla_optimal_tokens)

per_device_batch_size = 8
num_devices = torch.cuda.device_count()
if num_devices == 0:
    print("No GPU found, assuming CPU is used")
    num_devices = 1

total_batch_size = num_devices * per_device_batch_size
num_tokens_per_batch = total_batch_size * context_len
print("num_tokens_per_batch =", num_tokens_per_batch)
chinchilla_optimal_steps = chinchilla_optimal_tokens // num_tokens_per_batch

print("Loading dataset")

if dataset_name == "c4":
    c4_data_files = {
            "train": "en/*train*",
            "validation": "en/*validation*",
            }
    raw_datasets = datasets.load_dataset(
        "allenai/c4", 
        cache_dir=os.path.join(data_root_dir, "c4_cache"),
        data_files=c4_data_files,
        )
elif dataset_name == "wikitex":
    raw_datasets = datasets.load_dataset(
            "wikitext", 
            "wikitext-103-v1",
            cache_dir=os.path.join(data_root_dir, "wikitext_cache"),
            )
else:
    raise ValueError("Unknown dataset name")

text_column_name = "text"
column_names = raw_datasets["train"].column_names
assert column_names[0] == text_column_name

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

#p(raw_datasets)
#sys.exit(0)

def get_processed_dataset_cache_filenames(suffix):
	return {
		k: os.path.join(data_root_dir, f"{dataset_name}_processed", f"{k}_{suffix}.cache")
		for k in raw_datasets
	}

print("Tokenizing")
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
	remove_columns=column_names,
	load_from_cache_file=True,
	cache_file_names=get_processed_dataset_cache_filenames("tokenized"),
	num_proc=num_processing_workers,
    )


print("Preparing batches")
def group_texts(examples):
	# Concatenate all texts.
	concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	block_size = context_len
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
	# customize this part to your needs.
	if total_length >= block_size:
		total_length = (total_length // block_size) * block_size
	# Split by chunks of max_len.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
	}
	result["labels"] = result["input_ids"].copy()
	return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

lm_datasets = tokenized_datasets.map(
	group_texts,
	batched=True,
	load_from_cache_file=True,
	cache_file_names=get_processed_dataset_cache_filenames("grouped"),
	num_proc=num_processing_workers,
)

print("lm_datasets:", lm_datasets)

print("Creating trainer")
def preprocess_logits_for_metrics(logits, labels):
	if isinstance(logits, tuple):
		# Depending on the model and config, logits may contain extra tensors,
		# like past_key_values, but logits always come first
		logits = logits[0]
	return logits.argmax(dim=-1)

training_args = transformers.TrainingArguments(
    output_dir="/mnt/disk2/training/gpt2_c4",
    evaluation_strategy="steps",
    eval_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0,
    max_steps=2000,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    log_level="info",
    save_strategy="steps",
    save_steps=500,
    )

class MyTrainer(transformers.Trainer):
	#def _remove_unused_columns(self, dataset, description):
	#	print("Not removing unused columns!")
	#	return dataset
		
	def training_step(self, model, inputs):
		print("training_step inputs:", list(inputs.keys()))
		super().training_step(model, inputs)

	def compute_loss(self, model, inputs, return_outputs=False):
		print("compute_loss inputs:", list(inputs.keys()))
		#outputs = model(**inputs)
		#import pdb; pdb.set_trace()
		#print("Getting these outputs:", outputs)
		return super().compute_loss(model, inputs, return_outputs)

#trainer = transformers.Trainer(
trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
		data_collator=transformers.default_data_collator,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics,
		eval_dataset=None,
        #eval_dataset=lm_datasets["validation"], # TODO
        #tokenizer=tokenizer,
        #compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        #callbacks: Optional[List[TrainerCallback]] = None,
        #optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        )

print("Training...")
trainer.train()
