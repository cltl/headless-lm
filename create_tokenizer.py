"""
TO DO:
- add UniGram Tokenizer
- check BPE Tokenizer
"""

import click
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import json
import os

special_tokens_WP = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]


def get_training_corpus(dataset, ds_config):
    dataset = load_dataset(dataset, ds_config, split="train")

    def dataloader():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    return dataloader


def train_wp_tokenizer(vocab_size, dataloader):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens_WP
    )
    tokenizer.train_from_iterator(dataloader(), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return tokenizer


def train_bpe_tokenizer(vocab_size, dataloader):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|endoftext|>"]
    )
    tokenizer.train_from_iterator(dataloader(), trainer=trainer)
    return tokenizer


@click.command()
@click.option("-c", "--config")
@click.option("--dataset_name", default="BabyLM-strict-small")
@click.option("--dataset", default="cambridge-climb/BabyLM")
@click.option("--dataset_config", default="strict_small")
@click.option("--tokenizer_type", default="WordPiece")
@click.option("--vocab_size", default=25000)
@click.option("--save_dir", default="tokenizers")
def train_tokenizer(
    config,
    dataset_name,
    dataset,
    dataset_config,
    tokenizer_type,
    vocab_size,
    save_dir,
):
    if config is not None:
        with open(config) as f:
            cfg = json.load(f)
        dataset_name = cfg.get("dataset_name", "BabyLM-strict-small")
        dataset = cfg.get("dataset", "cambridge-climb/BabyLM")
        tokenizer_type = cfg.get("tokenizer_type", "WordPiece")
        vocab_size = cfg.get("vocab_size", 25000)
        dataset_config = cfg.get("dataset_config", "strict_small")
        save_dir = cfg.get("save_dir", "tokenizers")

    dataloader = get_training_corpus(dataset, dataset_config)
    if tokenizer_type == "WordPiece":
        tokenizer = train_wp_tokenizer(vocab_size, dataloader)
    elif tokenizer_type == "BPE":
        tokenizer = train_bpe_tokenizer(vocab_size, dataloader)
    else:
        raise NotImplementedError
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, f"{dataset_name}_{tokenizer_type}.json"))


if __name__ == "__main__":
    train_tokenizer()
