from engine.tasks.benchmark.glue import GlueBenchmark
from engine.lit.lightning_module import TaskTrainer
from pytorch_lightning import seed_everything
import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
os.environ['TOKENIZERS_PARALLELISM'] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("--ckpt_path")
parser.add_argument("--run_name")

args = parser.parse_args()

config = {}
if args.config is not None:
    config = json.load(open(args.config, "rb"))
logger.info(f"task config:\n{config}")

batch_size = int(config.get("batch_size", 32))
mode = config.get("mode", 'mlm')
learning_rate = config.get("learning_rate", 1e-5)
deterministic = config.get("deterministic", False)
seed = int(config.get("seed", 42))
seed_everything(seed, workers=True)

model_ckpt = args.ckpt_path
run_name = args.run_name

task_trainer = TaskTrainer.load_from_checkpoint(model_ckpt, map_location='cuda', weights_only=False)

tokenizer = task_trainer.task.tokenizer
tokenizer.pad_token = tokenizer.eos_token
if mode == "mlm":
    model = task_trainer.task.mlm_model
    backbone = 'mlm'
else: 
    model = task_trainer.task.lm_model

backbone = model



def main():
  GlueBenchmark(tokenizer, backbone, logger='wandb', logger_args={'project': 'GLUE'+run_name}, train_batch_size=batch_size, accumulate_grad_batches=1,
        learning_rate=learning_rate, deterministic=deterministic, num_workers=17
    )

if __name__ =='__main__':
  main()

