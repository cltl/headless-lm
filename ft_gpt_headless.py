from engine.data import DataModule
from engine.tasks.pretraining import GptHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
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
parser.add_argument("c", "--config")
parser.add_argument("j", "--job_config")

args = parser.parse_args()

config = {}
if args.config is not None:
    config = json.load(open(args.config, "rb"))
logger.info(f"task config:\n{config}")

job_config = {}
if args.job_config is not None:
    job_config = json.load(open(args.job_config, "rb"))
logger.info(f"job config:\n{job_config}")

dataset = job_config["dataset"]
num_gpus = int(job_config.get("num_gpus", 1))
num_workers = int(job_config.get("num_workers", 1))
ckpt_path = job_config.get("ckpt_path", None)
accu_grad_batches = int(job_config.get("accu_grad_batches", 1))
gpu_bs = int(job_config.get("gpu_bs", 16))
run_name = job_config.get("run_name", "test")
precision = job_config.get("precision", "16-mixed")
ckpt_every = job_config.get("ckpt_every", 50)
ckpt_save_dir = job_config.get("ckpt_save_dir", "ckpts")
mode = job_config.get("mode", "ft")
seed = int(job_config.get("seed", 57))


torch.set_float32_matmul_precision("high")

effective_bs = accu_grad_batches * num_gpus * gpu_bs
logger.info(
    f"Effective batch size is: {effective_bs}, based on device batch size, number of gpus and accumulation factor"
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
    num_workers=num_workers,
)

task_trainer = TaskTrainer.load_from_checkpoint(ckpt_path, map_location="cuda")

tokenizer = task_trainer.task.tokenizer
lm_model = task_trainer.task.lm_model
if mode == "probe":
    lm_model.gpt_neox.requires_grad_(False)

vocab_len, hs = lm_model.gpt_neox.get_input_embeddings().weight.shape

lm_model.embed_out = torch.nn.Linear(hs, vocab_len, bias=False)
lm_model.embed_out.weight.data = lm_model.get_input_embeddings().weight.data.clone()
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
        every_n_train_steps=1000, dirpath=f"{ckpt_save_dir}/{run_name}", save_top_k=1
    ),
]

trainer.fit(
    datamodule,
    precision=precision,
    accumulate_grad_batches=accu_grad_batches,
    callbacks=checkpoints,
    limit_val_batches=10,
    val_check_interval=2500,
    gradient_clip_val=1.0,
    benchmark=True,
    default_root_dir=f"{ckpt_save_dir}/{run_name}",
)
