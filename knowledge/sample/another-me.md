# another-me project

A text twin of its owner: a small open model that replies the way the owner would.

## Why it exists

The project is a hands-on showcase of LLM post-training. It also demonstrates a knowledge
base, because a fine-tuned adapter learns tone well but memorises facts badly.

## How it works

The writing style comes from a LoRA adapter trained on question and answer pairs that the owner wrote
themself. The base model is Qwen3 4B Instruct. Facts come from a folder of documents the
model can consult through tools.

## Knowledge modes

There are two ways the model can look things up, and only one is attached per run.
In files mode the model lists the documents and reads the ones it chooses.
In search mode the model sends a query and receives ranked passages.

## Training

Training runs on a Modal GPU with PyTorch and PEFT. Modal runs need a token.
