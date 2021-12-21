import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_ENRU_NAME = "Helsinki-NLP/opus-mt-en-ru"
BASE_RUEN_NAME = "Helsinki-NLP/opus-mt-ru-en"

BASE_ENRU_DIR = "./base_enru"
BASE_RUEN_DIR = "./base_ruen"

def create_base_model(name: str, directory: str) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model.save_pretrained(save_directory=directory)
    tokenizer.save_pretrained(save_directory=directory)
    
parser = argparse.ArgumentParser()

parser.add_argument("--create_enru", action="store_true", default=False)
parser.add_argument("--create_ruen", action="store_true", default=False)

args = parser.parse_args()

if args.create_enru:
    print("Generating en-ru base model...")
    create_base_model(name=BASE_ENRU_NAME, directory=BASE_ENRU_DIR)

if args.create_ruen:
    print("Generating ru-en base model...")
    create_base_model(name=BASE_RUEN_NAME, directory=BASE_RUEN_DIR)
