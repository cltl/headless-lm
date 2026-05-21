import argparse
import json
import logging
import os
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, AutoModel

from engine.tasks.benchmark.glue import GlueBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-m", "--model_id")
parser.add_argument("-t", "--tokenizer")
parser.add_argument("--run_name")

args = parser.parse_args()

config = {}
if args.config is not None:
    config = json.load(open(args.config, "rb"))
logger.info(f"task config:\n{config}")

batch_size = int(config.get("batch_size", 32))
mode = config.get("mode", "mlm")
learning_rate = config.get("learning_rate", 1e-5)
deterministic = config.get("deterministic", False)
seed = int(config.get("seed", 42))
seed_everything(seed, workers=True)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})
mlm_model = AutoModel.from_pretrained(args.model_id)

backbone = mlm_model

GlueBenchmark(
    tokenizer,
    backbone,
    logger="wandb",
    logger_args={"project": "GLUE" + args.run_name},
    train_batch_size=batch_size,
    accumulate_grad_batches=1,
    learning_rate=learning_rate,
    deterministic=deterministic,
    num_workers=17,
)

# GlueBenchmark(
#     tokenizer,
#     backbone,
#     logger="wandb",
#     logger_args={"project": "GLUE"},
#     train_batch_size=batch_size,
#     accumulate_grad_batches=1,
#     learning_rate=learning_rate,
#     weighted_ce=True,
#     weight_decay=0.01,
#     shuffle=True,
# )
