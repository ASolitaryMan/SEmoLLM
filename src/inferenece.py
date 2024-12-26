import os
import numpy as np
import torch
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, BertTokenizer, LlamaTokenizer

from accelerate import Accelerator
import argparse

from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
from copy import deepcopy
from lightning_fabric.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from bert_score import score
import torch.distributed as dist

from model_llama2 import CaptionLlama, CaptionLlamaNpro
import datetime
from torch.utils.data import WeightedRandomSampler
from dataloader import DataCollatorCTCWithPadding, CaptionDataset

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def get_loaders(args):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_path, return_attention_mask=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.llm_path, use_fast=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, tokenizer=tokenizer, padding="longest", first_stage=False, max_length=args.audio_maxlen, max_length_text=args.text_maxlen)

    test_dataset = CaptionDataset(args.test_src)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator,num_workers=args.num_workers)
    return test_dataloader



def valid_step(accelerator, model, dataloader, tokenizer, epoch):
    
    model.eval()

    batch_losses = 0.0
    all_infer_caption = []
    all_label_caption = []

    for step, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Validing'):
        input_values, attention_mask, text_input_values, text_attention_mask, texts = data["input_values"], data["attention_mask"], data["text_id"].input_ids, data["text_id"].attention_mask, data['caption']
        
        with torch.no_grad():
            outputs = model(input_values=input_values, attention_mask=attention_mask, texts=texts)
            loss = outputs.loss

            if accelerator.state.num_processes > 1:
                loss = accelerator.gather_for_metrics(loss.detach()).sum()
            
            if accelerator.state.num_processes > 1:
                generated_tokens = model.module.generate(input_values=input_values, attention_mask=attention_mask)
                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu()

                text_input_values = accelerator.pad_across_processes(text_input_values, dim=1, pad_index=tokenizer.pad_token_id)
                all_texts = accelerator.gather_for_metrics((text_input_values)).cpu()

                all_captions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                all_texts = tokenizer.batch_decode(all_texts, skip_special_tokens=True)
                all_infer_caption += all_captions
                all_label_caption += all_texts
                
            else:
                captions = model.generate(input_values=input_values, attention_mask=attention_mask)
                all_captions = tokenizer.batch_decode(captions, skip_special_tokens=True)
                all_texts = tokenizer.batch_decode(text_input_values, skip_special_tokens=True)
                all_infer_caption += all_captions
                all_label_caption += all_texts
        
            batch_losses += loss.item()

    if accelerator.is_main_process:
        for i in range(len(all_label_caption)):
            print(all_label_caption[i], all_infer_caption[i])


def get_current_time():

    # 获取当前时间
    current_time = datetime.datetime.now()

    # 将时间格式化为年月日时分秒格式
    formatted_time_a = current_time.strftime("%Y-%m-%d-%H:%M:%S")
    return formatted_time_a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--lora', action='store_true', default=False, help='whether use debug to limit samples')

    ## Params for training
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, metavar='nw', help='number of workers')
    parser.add_argument('--seed', type=int, default=1234, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--audio_maxlen', type=int, default=12, help='the max len of audio')
    parser.add_argument('--text_maxlen', type=int, default=64, help='the max len of audio')

    parser.add_argument('--test_src', type=str, default="/home/lqf/workspace/SEmoLLM/test.scp", help='the path of test_src')


    parser.add_argument('--wavlm_path', type=str, default="/home/lqf/workspace/wavlm-large", help='the path of wavlm_path')
    parser.add_argument('--bert_path', type=str, default="/home/lqf/workspace/bert-base-uncased", help='the path of bert_path')
    parser.add_argument('--llm_path', type=str, default="/home/lqf/workspace/vicuna-7b-v1.5", help='the path of llm_path')
    parser.add_argument('--ckpt_path', type=str, default="/home/lqf/workspace/audiocaption/phase2_ckpt/2024-09-02-15:00:58/43.pth", help='the path of Qformer')
    
    args = parser.parse_args()

    seed_everything(seed=args.seed)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100

    accelerator.print (f'====== Reading Data =======')
 
    test_loader = get_loaders(args)  
    
    
    accelerator.print (f'====== Training and Evaluation =======')

    accelerator.print (f'Step1: build model (each folder has its own model)')

    model = CaptionLlamaNpro(args.wavlm_path, args.bert_path, args.llm_path, args.ckpt_path, num_query_token=32, lora=args.lora)

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

    model.load_state_dict(checkpoint, strict=False)

    model, test_loader = accelerator.prepare(model, test_loader)
    
    device = accelerator.device

    accelerator.print (f'Step2: training (multiple epoches)')

    #phase two
    tokenizer = LlamaTokenizer.from_pretrained(args.llm_path, use_fast=False)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token


    tokenizer.padding_side = "right"

    valid_step(accelerator, model, test_loader, tokenizer, 43)

    #TODO metric