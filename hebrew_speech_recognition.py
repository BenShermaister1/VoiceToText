import glob
import os
from functools import partial
import datasets
import codecs

LANGS = [
    "he"
]
VERSION = datasets.Version("0.0.1")


class PublicSpeech(datasets.GeneratorBasedBuilder):
    """Public Speech dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=lang, version=VERSION, description=f"Public Speech {lang} dataset")
        for lang in LANGS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16000),
                    "sentence": datasets.Value("string"),
                }
            ),
            supervised_keys=("audio", "sentence"),

            homepage="https://huggingface.co/datasets/BenShermaister/VTTFPBS",
            citation="TODO",
        )

    def _split_generators(self, dl_manager):
        downloader = partial(
            lambda split: dl_manager.download_and_extract(f"data/{self.config.name}/{split}.tar.gz"),
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"root_path": downloader("train"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"root_path": downloader("dev"), "split": "dev"},
            ),
        ]

    def _generate_examples(self, root_path, split):
        split_path = os.path.join(root_path, split)
        for wav in glob.glob(split_path + "/*.wav"):
            uid = os.path.splitext(os.path.basename(wav))[0]
            txt_file = os.path.join(split_path, f"{uid}.txt")
            try:
                with codecs.open(txt_file, "r", encoding="utf-8", errors="replace") as fin:
                    text = fin.read()
            except UnicodeDecodeError as e:
                print(f"Error reading {txt_file}: {e}")
                with open(txt_file, "rb") as fin:
                    byte_sequence = fin.read()
                text = byte_sequence.decode("utf-8", errors="replace")

            example = {
                "audio": wav,
                "sentence": text,
            }
            yield uid, example
