from transformers import Wav2Vec2FeatureExtractor
from transformers import RobertaTokenizer,BertTokenizer, LlamaTokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import soundfile as sf
import json
import librosa

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: BertTokenizer
    padding: Union[bool, str] = "max_length"
    first_stage: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_text: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, text_input = [], []
        sample_rate = self.feature_extractor.sampling_rate

        for feature in features:
            input_features.append({"input_values": feature[0]})
            text_input.append(feature[1])
        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length * sample_rate,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            return_tensors="pt",
        )
        
        if self.first_stage:
            batch["text_id"] = self.tokenizer(
                text_input,
                padding="max_length", #这里需要注意
                truncation=True,
                max_length=self.max_length_text,
                return_tensors="pt",)
        else:
            batch["text_id"] = self.tokenizer(
                text_input,
                padding="longest", #这里需要注意
                truncation=True,
                max_length=self.max_length_text,
                return_tensors="pt",)
        
        # captions = self.tokenizer.batch_decode(batch["text_id"]['input_ids'], skip_special_tokens=False)
        # print(captions)
        # print(self.tokenizer.eos_token)
        if not self.first_stage:
            batch["caption"] = text_input
        #     batch["caption"] = self.tokenizer(
        #     [t + self.tokenizer.eos_token for t in text_input],
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_length_text,
        # )

        return batch

def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines    

class CaptionDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []

        for line in all_lines:
            tmp = line.strip().split("\n")[0].split("\t")
            self.wav_list.append(tmp[-1])
            self.label.append(tmp[0])
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        assert (sr == 16000)
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
        lab = self.label[index]
    
        return torch.FloatTensor(wave), lab

    def __len__(self):
        return len(self.label)
    
if __name__ == "__main__":
    pass