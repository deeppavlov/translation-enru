import argparse
from functools import partial

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm

DEFAULT_ENRU_MODEL = "DeepPavlov/marianmt-tatoeba-enru"
DEFAULT_RUEN_MODEL = "DeepPavlov/marianmt-tatoeba-ruen"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_RETURN_SEQUNCES = 1
DEFAULT_NUM_BEAMS = 3
DEFAULT_OUTPUT = "generated_translations.txt"


def resize(hypotheses, num_return_sequences):
    output = []
    for i in range(0, len(hypotheses), num_return_sequences):
        output.append(hypotheses[i: i + num_return_sequences])
    return output


def batch_beam_translate(sentence_batch, tokenizer, model, num_beams, num_return_sequences):
    batch_tokens = tokenizer(sentence_batch, return_tensors="pt", padding=True)
    outputs = model.generate(**batch_tokens, num_beams=num_beams, num_return_sequences=num_return_sequences)
    hypotheses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return resize(hypotheses, num_return_sequences)


def translate_all(sentences, translate_fn, batch_size):
    output = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        output.extend(translate_fn(sentences[i: i + batch_size]))
    return output


def write_beam_translations(path, hypotheses):
    with open(path, "w") as file:
        for beam in hypotheses:
            file.write("\t".join(beam) +  "\n")

            
def read_sentences(path):
    sentences = []
    with open(path) as file:
        for line in file:
            sentences.append(line.strip("\n"))
    return sentences


def load_model_and_tokenizer(name_or_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return model, tokenizer


parser = argparse.ArgumentParser()

parser.add_argument("--sentences_path", type=str, required=True)
parser.add_argument("--direction", type=str, required=True)
parser.add_argument("--model", type=str)
parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT)
parser.add_argument("--num_return_sequences", type=int, default=DEFAULT_NUM_RETURN_SEQUNCES)
parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

args = parser.parse_args()

assert args.direction in ("enru", "ruen")

if not args.model:
    if args.direction == "enru":
        model, tokenizer = load_model_and_tokenizer(DEFAULT_ENRU_MODEL)
    else:
        model, tokenizer = load_model_and_tokenizer(DEFAULT_RUEN_MODEL)
else:
    model, tokenizer = load_model_and_tokenizer(args.model)


translate_fn = partial(
    batch_beam_translate,
    tokenizer=tokenizer,
    model=model,
    num_beams=args.num_beams,
    num_return_sequences=args.num_return_sequences,
)

sentences_to_translate = read_sentences(args.sentences_path)
translations = translate_all(sentences_to_translate, translate_fn, args.batch_size)
write_beam_translations(args.output_file, translations)
