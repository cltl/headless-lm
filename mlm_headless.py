import json
from engine.data import DataModule
from engine.tasks.pretraining import MlmHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import argparse
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("CPU count: ", psutil.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-j", "--job_config")
parser.add_argument("--num_gpus")
parser.add_argument("--num_workers")
parser.add_argument("--accu_grad_batches")
parser.add_argument("--gpu_bs")
parser.add_argument("--dataset")
parser.add_argument("--hf_tokenizer")
parser.add_argument("--run_name")
parser.add_argument("--hf_path")
parser.add_argument("--accelerator", default="hf")
parser.add_argument("--precision", default="16-mixed")
parser.add_argument("--ckpt_path", nargs="?", const=None, type=str)
parser.add_argument("--saved_ckpt_path")
parser.add_argument("--ckpt_every", default=10000)

args = parser.parse_args()

config = args.config
model_max_seq_len = args.config.pop("max_seq_len", 128)

job_config = {}
if args.job_config is not None:
    job_config = json.load(open(args.job_config, "rb"))
num_gpus = int(job_config.get("num_gpus", args.num_gpus))
num_workers = int(job_config.get("num_workers", args.num_workers))
ckpt_path = job_config.get("ckpt_path", args.ckpt_path)
accu_grad_batches = int(job_config.get(
    "accu_grad_batches", args.accu_grad_batches))
gpu_bs = int(job_config.get("gpu_bs", args.gpu_bs))
dataset = job_config.get("dataset", args.dataset)
hf_tokenizer = job_config.get("hf_tokenizer", args.hf_tokenizer)
run_name = job_config.get("run_name", args.run_name)
hf_path = job_config.get("hf_path", args.hf_path)
accelerator = job_config.get("accelerator", args.accelerator)
precision = job_config.get("precision", args.precision)
ckpt_every = job_config.get("ckpt_every", args.ckpt_every)
saved_ckpt_path = job_config.get("saved_ckpt_path", args.saved_ckpt_path)

if accelerator == "xformers":
    from engine.models.xformers.efficient_bert import BertForMaskedLM
elif accelerator == "flash_attention":
    from engine.models.flash_attention.efficient_bert import BertForMaskedLM

    torch.set_float32_matmul_precision("medium")
elif accelerator == "hf":
    from transformers import BertForMaskedLM
else:
    raise NotImplementedError(
        f"Unknown accelerator {accelerator}. Please pick between 'hf', 'flash_attention', 'xformers'."
    )

global_bs = accu_grad_batches * num_gpus * gpu_bs
logger.info(
    f"Global batch size is: {global_bs}, based on selected batch size, number of gpus and accumulation factor"
)


datamodule = DataModule.from_datasets(
    dataset,
    train_batch_size=gpu_bs,
    infer_batch_size=gpu_bs,
    split_names=["train", "validation", "test"],
    from_disk=True,
    num_workers=num_workers,
)


tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
lm_config = AutoConfig.from_pretrained(hf_path)

torch.set_float32_matmul_precision("medium")

lm_config.vocab_size = len(tokenizer.vocab)
tokenizer.mask_token_id = 1
lm_config.max_position_embeddings = model_max_seq_len
lm_model = BertForMaskedLM(lm_config)


task = MlmHeadlessPretraining(tokenizer, lm_model, config=config)

version_name = run_name
trainer = TaskTrainer(
    task, logger="wandb", logger_args={"project": "MLM-headless", "name": version_name}
)

checkpoints = [
    ModelCheckpoint(
        every_n_train_steps=ckpt_every,
        dirpath=f"{saved_ckpt_path}/{version_name}",
        save_top_k=-1,
    ),
    # ModelCheckpoint(
    #     every_n_train_steps=1000,
    #     dirpath=f"{saved_ckpt_path}/{version_name}_last",
    #     save_top_k=1,
    # ),
]

trainer.fit(
    datamodule,
    # num_nodes=num_nodes,  # accelerator and number of devices set to auto
    precision=precision,
    accumulate_grad_batches=accu_grad_batches,
    callbacks=checkpoints,
    limit_val_batches=10,
    val_check_interval=0.1,
    gradient_clip_val=1.0,
    benchmark=True,
    default_root_dir=f"{saved_ckpt_path}/{version_name}",
    ckpt_path=ckpt_path,
)
