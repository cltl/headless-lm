import os
from engine.data import DataModule
from engine.tasks.pretraining import GptHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import log
import argparse
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("CPU count: ", psutil.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-j", "--job_config")
parser.add_argument("--num_nodes")
parser.add_argument("--global_bs")
parser.add_argument("--gpu_bs")
parser.add_argument("--dataset")
parser.add_argument("--mode", default='ft')
parser.add_argument("--run_name")
parser.add_argument("--precision", default="16-mixed")
parser.add_argument('--ckpt_path', nargs='?', const=None, type=str)
parser.add_argument('--saved_ckpt_path')
parser.add_argument("--ckpt_every", default=2500)

args = parser.parse_args()

config = json.load(open(args.config, "rb"))
model_max_seq_len = int(config.get("model_max_seq_len", args.model_max_seq_len)

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
mode = job_config.get("mode", args.mode)
hf_tokenizer = job_config.get("hf_tokenizer", args.hf_tokenizer)
run_name = job_config.get("run_name", args.run_name)
hf_path = job_config.get("hf_path", args.hf_path)
accelerator = job_config.get("accelerator", args.accelerator)
precision = job_config.get("precision", args.precision)
ckpt_every = job_config.get("ckpt_every", args.ckpt_every)
saved_ckpt_path = job_config.get("saved_ckpt_path", args.saved_ckpt_path)

if "A100" in torch.cuda.get_device_name():
  torch.set_float32_matmul_precision('high')

global_bs = accu_grad_batches * num_gpus * gpu_bs
logger.info(
    f"Global batch size is: {global_bs}, based on selected batch size, number of gpus and accumulation factor"
)

# gpus_by_node = torch.cuda.device_count()
#
# if ((gpus_by_node * num_nodes) % global_bs) == 0:
#   raise argparse.ArgumentError(f"Requested a batch size of {global_bs} on {gpu_bs}x{gpus_by_node} GPUs : not a multiple!")
# accu_grad_batches = global_bs // (gpus_by_node * num_nodes * gpu_bs)
# print(f"Grad. accumulating factor: {accu_grad_batches}")


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
if mode=="probe":
  lm_model.gpt_neox.requires_grad_(False)

vocab_len, hs = lm_model.gpt_neox.get_input_embeddings().weight.shape

lm_model.embed_out = torch.nn.Linear(hs, vocab_len, bias=False)
lm_model.embed_out.weight.data = lm_model.get_input_embeddings().weight.data.clone()
logger.info(lm_model)



task = GptHeadlessPretraining(
    tokenizer, lm_model, config = config
)

version_name = run_name
trainer = TaskTrainer(
    task, logger="wandb", logger_args={"project": "ft_gpt_headless", "name": version_name}
)

checkpoints = [
  ModelCheckpoint(every_n_train_steps=ckpt_every, dirpath=f'{saved_ckpt_path}/{version_name}', save_top_k=-1),
  # ModelCheckpoint(every_n_train_steps=1000, dirpath=f'{saved_ckpt_path}/{version_name}', save_top_k=1)
]

trainer.fit(
  datamodule,
  # num_nodes=num_nodes,
  precision=precision,
  accumulate_grad_batches=accu_grad_batches,
  callbacks=checkpoints,
  limit_val_batches=10,
  val_check_interval=2500,
  gradient_clip_val=1.0,
  benchmark=True,
  default_root_dir=f'{saved_ckpt_path}/{version_name}',
)
