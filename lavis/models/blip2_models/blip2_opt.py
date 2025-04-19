"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers

# Import the load_pretrained_model function
# from mujoco_vc.model_loading import load_pretrained_model, fuse_embeddings_flare
from stable_control_representations.cortexbench.mujoco_vc.src.mujoco_vc.model_loading import load_pretrained_model, fuse_embeddings_flare
from stable_control_representations.cortexbench.mujoco_vc.visual_imitation.hydra_launcher import configure_jobs
# import os
import os
from omegaconf import OmegaConf


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()
        self.sd_model_config = configure_jobs("stable_control_representations/cortexbench/mujoco_vc/visual_imitation/config/Metaworld_BC_config.yaml")
        #("LAVIS/stable-control-representations/cortexbench/mujoco_vc/visual_imitation/config/Metaworld_BC_config.yaml")
        # "/opt/data/borghei/LAVIS/stable-control-representations/cortexbench/mujoco_vc/visual_imitation/config/Metaworld_BC_config.yaml"
        


        base_dir = "/opt/data/borghei/LAVIS"  # Adjust this to your workspace root
        model_config_path = os.path.join(
            base_dir, 
            "stable_control_representations/vc_models/src/vc_models/conf/model/diffusion_sd_15_laion.yaml"
        )

        # Load model config directly
        model_config = OmegaConf.load(model_config_path)
        # Load Stable Diffusion model for feature extraction
        # print("model_config", model_config)
        self.sd_model, self.embedding_dim, self.transforms, self.metadata = load_pretrained_model(
            # self.sd_model_config["env_kwargs"]["embedding_config"]
            model_config
        )        
        for name, param in self.sd_model.named_parameters():
            param.requires_grad = False
        self.sd_model = self.sd_model.eval()
        self.sd_model.train = disabled_train
        logging.info("freeze stable diffusion model")
        self.diffusion_timesteps = self.sd_model_config["diffusion_timesteps"]

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            #  vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.dim_reducer = nn.Linear(
            self.embedding_dim, 768, bias=False  # Reduce from 286720 to 768
        ).to(torch.float16)  # Use half precision to save memory    

        self.opt_proj = nn.Linear(
            # self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
            #self.embedding_dim, self.opt_model.config.hidden_size
            768, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

    def forward(self, samples):
        image = samples["image"]

        # processed_images = torch.cat([self.transforms(img) for img in image])
        processed_images = torch.cat([self.transforms(img) for img in image])
        #Note the device for later use
        # device = processed_images.device

        with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )

            diffusion_prompt = samples["raw_caption"] 
            # Extract embeddings from SD model
            with torch.no_grad():# if not self.sd_model.training else torch.enable_grad():
                if diffusion_prompt is not None:
                    image_embeddings = self.sd_model(
                        processed_images,#processed_images.to(self.device) 
                        diffusion_prompt, 
                        self.diffusion_timesteps, 
                        #new_noise=True
                    )
                    print("diffusion_prompt", diffusion_prompt)
                else:
                    image_embeddings = self.sd_model(processed_images)#processed_images.to(self.device)
                    print("without diffusion_prompt")

                # save embedding in RAM and free up GPU memory
                image_embeddings = image_embeddings.detach()#.to("cpu").data.numpy() 
        
        # Project SD embeddings to OPT dimensions
        image_embeddings = self.dim_reducer(image_embeddings)
        inputs_opt = self.opt_proj(image_embeddings)#(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(self.device)#.to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)#.to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(self.device).fill_(-100)#.to(image.device).fill_(-100)
        )
        print("empty_targets", empty_targets.shape)
        print("targets", targets.shape)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
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
        image = samples["image"]

        processed_images = torch.cat([self.transforms(img) for img in image])


        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            #     image.device
            # )

            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_output = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )
            diffusion_prompt = samples["raw_caption"] 
            # Extract embeddings from SD model
            with torch.no_grad():# if not self.sd_model.training else torch.enable_grad():
                if diffusion_prompt is not None:
                    image_embeddings = self.sd_model(
                        processed_images,#processed_images.to(self.device) 
                        diffusion_prompt, 
                        self.diffusion_timesteps, 
                        #new_noise=True
                    )
                    print("diffusion_prompt", diffusion_prompt)
                else:
                    image_embeddings = self.sd_model(processed_images)#processed_images.to(self.device)
                    print("without diffusion_prompt")

                # save embedding in RAM and free up GPU memory
                image_embeddings = image_embeddings.detach()#.to("cpu").data.numpy() 
            
            # Project SD embeddings to OPT dimensions
            # inputs_opt = self.opt_proj(query_output.last_hidden_state)
            image_embeddings = self.dim_reducer(image_embeddings)
            inputs_opt = self.opt_proj(image_embeddings)
            
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                self.device#image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)#.to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        processed_images = torch.cat([self.transforms(img) for img in image])

        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            #     image.device
            # )

            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_output = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )
            diffusion_prompt = samples["raw_caption"] 
            # Extract embeddings from SD model
            with torch.no_grad():# if not self.sd_model.training else torch.enable_grad():
                if diffusion_prompt is not None:
                    image_embeddings = self.sd_model(
                        processed_images,#processed_images.to(self.device) 
                        diffusion_prompt, 
                        self.diffusion_timesteps, 
                        #new_noise=True
                    )
                    print("diffusion_prompt", diffusion_prompt)
                else:
                    image_embeddings = self.sd_model(processed_images)#processed_images.to(self.device)
                    print("without diffusion_prompt")

                # save embedding in RAM and free up GPU memory
                image_embeddings = image_embeddings.detach()#.to("cpu").data.numpy() 
            
            # Project SD embeddings to OPT dimensions
            # inputs_opt = self.opt_proj(query_output.last_hidden_state)
           

            image_embeddings = self.dim_reducer(image_embeddings)
            inputs_opt = self.opt_proj(image_embeddings)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                self.device#image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)#.to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model