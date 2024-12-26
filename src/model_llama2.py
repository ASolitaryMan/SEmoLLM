from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WavLMPreTrainedModel, LlamaTokenizer
from transformers import BertTokenizer
import torch.nn as nn
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from Qformer import BertConfig, BertLMHeadModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from modeling_llama import LlamaForCausalLM
import logging
from peft import LoraConfig, TaskType, get_peft_model

class CaptionLlama(nn.Module):
    def __init__(self, wavlm_path, bert_path, vicuna_path, ckpt_path, num_query_token=32, max_txt_len=80, embed_dim=256, use_weighted_layer_sum=True, freeze_wavlm=True, lora=False, lora_rank=2, lora_alpha=8, lora_dropout=0.1):
        super(CaptionLlama, self).__init__()

        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        self.lora = lora
        self.bert_path = bert_path
        self.vicuna_path = vicuna_path
        self.max_txt_len = max_txt_len
        self.use_weighted_layer_sum = use_weighted_layer_sum

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm = self.wavlm.eval()
        
        num_layers = self.wavlm.config.num_hidden_layers + 1  # transformer layers + input embeddings
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.projector = nn.Linear(self.wavlm.config.hidden_size, self.wavlm.config.classifier_proj_size)
        
        self.dropout = nn.Dropout(self.wavlm.config.final_dropout)
        self.ln_audio = nn.LayerNorm(self.wavlm.config.classifier_proj_size)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.wavlm.config.classifier_proj_size
        )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_path, torch_dtype=torch.float16)


        # Decoder Only
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.vicuna_path, use_fast=False)
        # self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        
        self.llama_tokenizer.padding_side = "right"

        # self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        
        #TODO LORA
        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()


        # self.prompt = "Please describe the pitch, speed, and volume in the audio."
        
        self.prompt = "Please recognize the emotions contained in the speech based on the description of vocal features and the content within the audio."
            
    
    def init_Qformer(self, num_query_token, speech_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.bert_path)
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            self.bert_path, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens
    
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def forward(self, 
                input_values: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                texts: Optional[list] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = True,
                return_dict: Optional[bool] = True):

        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        
        with torch.no_grad():
            outputs = self.wavlm(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            # use_cache=True,
            return_dict=True,
        )

        input_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(input_llm.size()[:-1], dtype=torch.long).to(
            input_values.device
        )

        self.llama_tokenizer.padding_side = "right"

        text_output_tokens = self.llama_tokenizer(
            [t + self.llama_tokenizer.eos_token for t in texts],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(input_values.device)

        targets = text_output_tokens['input_ids'].masked_fill(
            text_output_tokens['input_ids'] == self.llama_tokenizer.pad_token_id, -100
        )
        # targets = targets[:,1:]

        # # do not apply loss to the text input (i.e., instruction)
        # for i, l in enumerate(input_part_targets_len):
        #     targets[i][:l] = -100

        ############################################################################################################################################
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt").to(input_values.device)
        batchsize = input_llm.shape[0] 
        prompts_id=prompt_tokens['input_ids'][:,1:].expand(batchsize, -1)
        prompts_embeds=self.llama_model.model.embed_tokens(prompts_id) if not self.lora else self.llama_model.model.model.embed_tokens(prompts_id)
        attns_prompt=prompt_tokens['attention_mask'][:,1:].expand(batchsize, -1)
        ############################################################################################################################################


        inputs_embeds = self.llama_model.model.embed_tokens(text_output_tokens['input_ids']) if not self.lora else self.llama_model.model.model.embed_tokens(text_output_tokens['input_ids'])
        inputs_embeds = torch.cat([input_llm, prompts_embeds, inputs_embeds], dim=1) #
        attention_mask = torch.cat([atts_llm, attns_prompt, text_output_tokens['attention_mask']], dim=1)

        empty_targets = (
            torch.ones([batchsize, atts_llm.shape[1] + attns_prompt.shape[1]], dtype=torch.long).to(input_values.device).fill_(-100)
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        outputs=self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
        )


        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        use_nucleus_sampling: Optional[bool] = True,
        num_beams: Optional[int] = 5,
        max_length: Optional[int] = 80,
        min_length: Optional[int] = 3,
        top_p: Optional[float] = 0.9,
        repetition_penalty: Optional[float] = 1.5,
    ):
        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)

        
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
        
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(input_values.device)

        ############################################################################################################################################
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt").to(input_values.device)
        batchsize = inputs_llm.shape[0] 
        prompts_id=prompt_tokens['input_ids'][:,1:].expand(batchsize, -1)
        prompts_embeds=self.llama_model.model.embed_tokens(prompts_id) if not self.lora else self.llama_model.model.model.embed_tokens(prompts_id)
        attns_prompt=prompt_tokens['attention_mask'][:,1:].expand(batchsize, -1)
        ############################################################################################################################################

        # batchsize = inputs_llm.shape[0]
        bos = torch.ones([batchsize, 1], dtype=torch.int64).to(input_values.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        bos_att = atts_llm[:,:1]

        inputs_embeds = torch.cat([inputs_llm, prompts_embeds, bos_embeds], dim=1)
        atten_mask = torch.cat([atts_llm, attns_prompt, bos_att], dim=1)

        outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=atten_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=1,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=1,
                num_return_sequences=1,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                pad_token_id=self.llama_tokenizer.pad_token_id,
            )

        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)

        return outputs
    @torch.no_grad()
    def generate2(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        use_nucleus_sampling: Optional[bool] = True,
        num_beams: Optional[int] = 5,
        max_length: Optional[int] = 30,
        min_length: Optional[int] = 3,
        top_p: Optional[float] = 0.9,
        repetition_penalty: Optional[float] = 1.5,
    ):
        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)

        
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
        
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(input_values.device)

        batchsize = inputs_llm.shape[0]
        bos = torch.ones([batchsize, 1], dtype=torch.int64).to(input_values.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_att = atts_llm[:,:1]

        inputs_embeds = torch.cat([inputs_llm, bos_embeds], dim=1)
        atten_mask = torch.cat([atts_llm, bos_att], dim=1)

        outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=atten_mask,
                max_new_tokens=80,
                min_new_tokens=3,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                num_beams=5,
                repetition_penalty=10.0,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                #stopping_criteria=stopping_criteria,
                early_stopping=True,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )

        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)

        return outputs

class CaptionLlamaNpro(nn.Module):
    def __init__(self, wavlm_path, bert_path, vicuna_path, ckpt_path, num_query_token=32, max_txt_len=80, embed_dim=256, use_weighted_layer_sum=True, freeze_wavlm=True, lora=False, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
        super(CaptionLlamaNpro, self).__init__()

        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        self.lora = lora
        self.bert_path = bert_path
        self.vicuna_path = vicuna_path
        self.max_txt_len = max_txt_len
        self.use_weighted_layer_sum = use_weighted_layer_sum

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm = self.wavlm.eval()
        
        num_layers = self.wavlm.config.num_hidden_layers + 1  # transformer layers + input embeddings
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.projector = nn.Linear(self.wavlm.config.hidden_size, self.wavlm.config.classifier_proj_size)
        
        self.dropout = nn.Dropout(self.wavlm.config.final_dropout)
        self.ln_audio = nn.LayerNorm(self.wavlm.config.classifier_proj_size)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.wavlm.config.classifier_proj_size
        )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_path, torch_dtype=torch.float16)


        # Decoder Only
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.vicuna_path, use_fast=False)
        # self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        
        self.llama_tokenizer.padding_side = "right"

        # self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        
        #TODO LORA
        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()


        self.prompt = "Please describe the pitch, speed, and volume in the audio."
        
        #"Please recognize the emotions contained in the speech based on the description of vocal features and the content within the audio."
            
    
    def init_Qformer(self, num_query_token, speech_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.bert_path)
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            self.bert_path, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens
    
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def forward(self, 
                input_values: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                texts: Optional[list] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = True,
                return_dict: Optional[bool] = True):

        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        with torch.no_grad():
            outputs = self.wavlm(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            # use_cache=True,
            return_dict=True,
        )

        input_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(input_llm.size()[:-1], dtype=torch.long).to(
            input_values.device
        )

        self.llama_tokenizer.padding_side = "right"

        text_output_tokens = self.llama_tokenizer(
            [t + self.llama_tokenizer.eos_token for t in texts],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(input_values.device)

        targets = text_output_tokens['input_ids'].masked_fill(
            text_output_tokens['input_ids'] == self.llama_tokenizer.pad_token_id, -100
        )
        # targets = targets[:,1:]

        # # do not apply loss to the text input (i.e., instruction)
        # for i, l in enumerate(input_part_targets_len):
        #     targets[i][:l] = -100

        ############################################################################################################################################
        # prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt").to(input_values.device)
        batchsize = input_llm.shape[0] 
        # prompts_id=prompt_tokens['input_ids'][:,1:].expand(batchsize, -1)
        # prompts_embeds=self.llama_model.model.embed_tokens(prompts_id) if not self.lora else self.llama_model.model.model.embed_tokens(prompts_id)
        # attns_prompt=prompt_tokens['attention_mask'][:,1:].expand(batchsize, -1)
        # ############################################################################################################################################


        inputs_embeds = self.llama_model.model.embed_tokens(text_output_tokens['input_ids']) if not self.lora else self.llama_model.model.model.embed_tokens(text_output_tokens['input_ids'])
        # inputs_embeds = torch.cat([input_llm, prompts_embeds, inputs_embeds], dim=1) #
        # attention_mask = torch.cat([atts_llm, attns_prompt, text_output_tokens['attention_mask']], dim=1)
        inputs_embeds = torch.cat([input_llm, inputs_embeds], dim=1) #
        attention_mask = torch.cat([atts_llm, text_output_tokens['attention_mask']], dim=1)

        # empty_targets = (
        #     torch.ones([batchsize, atts_llm.shape[1] + attns_prompt.shape[1]], dtype=torch.long).to(input_values.device).fill_(-100)
        # )

        empty_targets = (
            torch.ones([batchsize, atts_llm.shape[1]], dtype=torch.long).to(input_values.device).fill_(-100)
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        outputs=self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
        )


        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        use_nucleus_sampling: Optional[bool] = True,
        num_beams: Optional[int] = 5,
        max_length: Optional[int] = 80,
        min_length: Optional[int] = 3,
        top_p: Optional[float] = 0.9,
        repetition_penalty: Optional[float] = 1.5,
    ):
        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)

        
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
        
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(input_values.device)

        ############################################################################################################################################
        # prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt").to(input_values.device)
        batchsize = inputs_llm.shape[0] 
        # prompts_id=prompt_tokens['input_ids'][:,1:].expand(batchsize, -1)
        # prompts_embeds=self.llama_model.model.embed_tokens(prompts_id) if not self.lora else self.llama_model.model.model.embed_tokens(prompts_id)
        # attns_prompt=prompt_tokens['attention_mask'][:,1:].expand(batchsize, -1)
        ############################################################################################################################################

        # batchsize = inputs_llm.shape[0]
        bos = torch.ones([batchsize, 1], dtype=torch.int64).to(input_values.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        bos_att = atts_llm[:,:1]

        # inputs_embeds = torch.cat([inputs_llm, prompts_embeds, bos_embeds], dim=1)
        # atten_mask = torch.cat([atts_llm, attns_prompt, bos_att], dim=1)

        inputs_embeds = torch.cat([inputs_llm, bos_embeds], dim=1)
        atten_mask = torch.cat([atts_llm, bos_att], dim=1)

        outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=atten_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=1,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=1,
                num_return_sequences=1,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                pad_token_id=self.llama_tokenizer.pad_token_id,
            )

        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)

        return outputs
    @torch.no_grad()
    def generate2(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        use_nucleus_sampling: Optional[bool] = True,
        num_beams: Optional[int] = 5,
        max_length: Optional[int] = 30,
        min_length: Optional[int] = 3,
        top_p: Optional[float] = 0.9,
        repetition_penalty: Optional[float] = 1.5,
    ):
        return_dict = return_dict if return_dict is not None else self.wavlm.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            hidden_states = self.projector(hidden_states)
        else:
            hidden_states = outputs[0]

        # hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        audio_embeds = self.ln_audio(hidden_states)

        
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
        
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(input_values.device)

        batchsize = inputs_llm.shape[0]
        bos = torch.ones([batchsize, 1], dtype=torch.int64).to(input_values.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_att = atts_llm[:,:1]

        inputs_embeds = torch.cat([inputs_llm, bos_embeds], dim=1)
        atten_mask = torch.cat([atts_llm, bos_att], dim=1)

        outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=atten_mask,
                max_new_tokens=80,
                min_new_tokens=3,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                num_beams=5,
                repetition_penalty=10.0,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                #stopping_criteria=stopping_criteria,
                early_stopping=True,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )

        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)

        return outputs    

    
if __name__ == "__main__":
    wavlm_path = "/home/lqf/workspace/wavlm-large"
    bert_path = "/home/lqf/workspace/bert-base-uncased"
    vicuna_path = "/home/lqf/workspace/Llama-2-7b-chat-hf"
    ckpt_path = "/home/lqf/workspace/audiocaption/src/session5/14.pth"

    CaptionLlama(wavlm_path, bert_path, vicuna_path, ckpt_path)
    pass
