from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WavLMPreTrainedModel
from transformers import BertTokenizer
import torch.nn as nn
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from Qformer import BertConfig, BertLMHeadModel
import torch.distributed as dist

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from dataclasses import dataclass


@dataclass
class BlipSimilarity(ModelOutput):
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None

@dataclass
class BlipIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models.

    image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2).
    itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,)

    """

    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None

@dataclass
class BlipOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[BlipSimilarity] = None

    intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

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

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

class CaptionQformer(nn.Module):
    def __init__(self, wavlm_path, bert_path, num_query_token, max_txt_len=32, embed_dim=256, use_weighted_layer_sum=True, freeze_wavlm=True):
        super(CaptionQformer, self).__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_path)
        self.bert_path = bert_path
        self.tokenizer = self.init_tokenizer(self.bert_path)
        self.use_weighted_layer_sum = use_weighted_layer_sum

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm = self.wavlm.eval()
        
        self.wavlm.freeze_feature_extractor()
        
        self.dropout = nn.Dropout(self.wavlm.config.final_dropout)
        num_layers = self.wavlm.config.num_hidden_layers + 1  # transformer layers + input embeddings
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.projector = nn.Linear(self.wavlm.config.hidden_size, self.wavlm.config.classifier_proj_size)
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.wavlm.config.classifier_proj_size, cross_attention_freq=2
        )#self.wavlm.config.output_hidden_size

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.audio_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        
        self.ln_audio = nn.LayerNorm(self.wavlm.config.classifier_proj_size)#self.wavlm.config.output_hidden_size

    def init_tokenizer(self, bert_path, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained(bert_path, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer 
    
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
            torch.zeros(1, num_query_token, encoder_config.hidden_size) # 1, 32, 1024
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


    def forward(self, 
                input_values: Optional[torch.Tensor],
                text_tokens_ids: Optional[torch.Tensor],
                text_tokens_attention_mask: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                label: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = True,
                return_dict: Optional[bool] = True):

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
            use_cache=True,
            return_dict=True,
        ) # batch, num_query, dim

        audio_feats = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )

        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(input_values.device)

        text_output = self.Qformer.bert(
            text_tokens_ids,
            attention_mask=text_tokens_attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        audio_feats_all = concat_all_gather(
            audio_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            audio_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), audio_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        #####################################
        # rank = dist.get_rank() #TODO
        # rank = 0
        #####################################
        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        bs = input_values.size(0)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            input_values.device
        )

        if label is not None:
            label = label.view(-1, 1)
            all_label = concat_all_gather(label)
            pos_idx = torch.eq(label, all_label.t()).float() 

            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

            # sim_targets = pos_idx

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
            loss_itc = (loss_t2i+loss_i2t)/2
        else:
            loss_itc = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2
        
        
        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens_ids)
        text_attention_mask_world = concat_all_gather(text_tokens_attention_mask)
        audio_embeds_world = all_gather_with_grad(audio_embeds)

        with torch.no_grad():
            if label is not None:
                mask = torch.eq(label, all_label.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)  

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)
        
        # select a negative image for each text
        audio_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens_ids, text_tokens_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens_attention_mask, text_tokens_attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            input_values.device
        )

        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        audio_embeds_all = torch.cat(
            [audio_embeds, audio_embeds_neg, audio_embeds], dim=0
        )  # pos, neg, pos
        audio_atts_all = torch.ones(audio_embeds_all.size()[:-1], dtype=torch.long).to(
            input_values.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=audio_embeds_all,
            encoder_attention_mask=audio_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(input_values.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

         ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            input_values.device
        )
        attention_mask = torch.cat([query_atts, text_tokens_attention_mask], dim=1)
        
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        # return BlipOutput(
        #     loss=loss_lm,
        #     loss_itc=loss_itc,
        #     loss_lm=loss_lm,
        # )

        # return BlipOutput(
        #     loss=loss_itc + loss_itm + loss_lm,
        #     loss_itc=loss_itc,
        #     loss_itm=loss_itm,
        #     loss_lm=loss_lm,
        # )

        return BlipOutput(
            loss=loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        use_nucleus_sampling: Optional[bool] = True,
        num_beams: Optional[int] = 3,
        max_length: Optional[int] = 90,
        min_length: Optional[int] = 3,
        top_p: Optional[float] = 0.9,
        repetition_penalty: Optional[float] =1.5,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
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

        # if not use_nucleus_sampling:
        #     audio_embeds = audio_embeds.repeat_interleave(num_beams, dim=0)
        # else:
        #     num_beams = 1
        
        audio_atts = self.wavlm._get_feature_vector_attention_mask(audio_embeds.shape[1], attention_mask).to(torch.long)

        model_kwargs = {
            "encoder_hidden_states": audio_embeds,
            "encoder_attention_mask": audio_atts,
        }

        input_ids = (
            torch.LongTensor(input_values.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(input_values.device)
        )
        
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs
        )
        # captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

