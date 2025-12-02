from engine.data import DataModule
from engine.tasks.pretraining import GptHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import json
import logging
import numpy as np
import random
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-j", "--job_config")

args = parser.parse_args()

config = {}
if args.config is not None:
    config = json.load(open(args.config, "rb"))
model_max_seq_len = config.get("max_seq_len", 2048)
logger.info(f"task config:\n{config}")

job_config = {}
if args.job_config is not None:
    job_config = json.load(open(args.job_config, "rb"))
logger.info(f"job config:\n{job_config}")

dataset = job_config["dataset"]
hf_tokenizer = job_config["hf_tokenizer"]
num_gpus = int(job_config.get("num_gpus", 1))
# num_workers = int(job_config.get("num_workers", 1))
ckpt_path = job_config.get("ckpt_path", None)
accu_grad_batches = int(job_config.get("accu_grad_batches", 1))
gpu_bs = int(job_config.get("gpu_bs", 16))
run_name = job_config.get("run_name", "test")
hf_path = job_config.get("hf_path", "google-bert/bert-base-uncased")
accelerator = job_config.get("accelerator", "hf")
precision = job_config.get("precision", "16-mixed")
ckpt_every = job_config.get("ckpt_every", 1000)
ckpt_save_dir = job_config.get("ckpt_save_dir", "ckpts")
seed = int(job_config.get("seed", 57))


if accelerator == "xformers":
    from engine.models.xformers.efficient_gpt_neox import GPTNeoXForCausalLM
elif accelerator == "flash_attention":
    from engine.models.flash_attention.efficient_gpt_neox import GPTNeoXForCausalLM
elif accelerator == "hf":
    from transformers import GPTNeoXForCausalLM
else:
    raise NotImplementedError(
        f"Unknown accelerator {accelerator}. Please pick between 'hf', 'flash_attention', 'xformers'."
    )

torch.set_float32_matmul_precision("high")

global_bs = accu_grad_batches * num_gpus * gpu_bs
logger.info(
    f"Global batch size is: {global_bs}, based on selected batch size, number of gpus and accumulation factor"
)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


datamodule = DataModule.from_datasets(
    dataset,
    train_batch_size=gpu_bs,
    infer_batch_size=gpu_bs,
    split_names=["train", "validation", "test"],
    from_disk=True,
    num_workers=0,
)


tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
lm_config = AutoConfig.from_pretrained(hf_path)

lm_config.max_position_embeddings = model_max_seq_len
lm_model = GPTNeoXForCausalLM(lm_config)
print(lm_model)


task = GptHeadlessPretraining(tokenizer, lm_model, config=config)

trainer = TaskTrainer(task, logger_args={"version": run_name})

checkpoints = [
    ModelCheckpoint(
        every_n_train_steps=ckpt_every,
        dirpath=f"{ckpt_save_dir}/{run_name}",
        save_top_k=-1,
    ),
    ModelCheckpoint(
        every_n_train_steps=1000,
        dirpath=f"{ckpt_save_dir}/{run_name}",
        save_top_k=1,
    ),
]

trainer.fit(
    datamodule,
    precision=precision,
    accumulate_grad_batches=accu_grad_batches,
    callbacks=checkpoints,
    limit_val_batches=10,
    val_check_interval=0.1,
    gradient_clip_val=1.0,
    benchmark=True,
    default_root_dir=f"{ckpt_save_dir}/{run_name}",
    ckpt_path=ckpt_path,
)
