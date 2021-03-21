import argparse
from argparse import ArgumentParser
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import math

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from nlp import load_metric
import string
from pathlib import Path
from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
from torch.utils.data import RandomSampler

import textwrap
from tqdm.auto import tqdm
from nlp import load_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def approx_match_score(prediction, ground_truth):
    answer = normalize_answer(prediction) 
    gt = normalize_answer(ground_truth)
    match = 0
    gt_words = gt.split(" ")
    for word in gt_words:
        if word in answer:
            match = 1
            return match
    return match

def calculate_scores(predictions, ground_truths):
    em_score = 0
    subset_match_score = 0
    
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        em_score +=  exact_match_score(prediction, ground_truth)
        subset_match_score += approx_match_score(prediction, ground_truth)
    
    em_score /= len(predictions)
    subset_match_score /= len(predictions)
    return em_score*100, subset_match_score*100

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
#         self.config = T5Config(hparams.model_name_or_path,dropout_rate=0.2)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
#         self.model.dropout_rate=0.2
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        self.em_score_list = []
        self.subset_score_list =[]
    
    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
            
            
    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.proc_rank <= 0
    
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
    )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    
    def _generative_step(self, batch) :
        
        t0 = time.time()
        
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])
            
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]  
    
        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=targets)
        em_score, subset_match_score = calculate_scores(preds, targets)
        
        self.em_score_list.append(em_score)
        self.subset_score_list.append(subset_match_score)
        
        em_score = torch.tensor(em_score,dtype=torch.float32)
        subset_match_score = torch.tensor(subset_match_score,dtype=torch.float32)
        
        base_metrics.update(em_score=em_score, subset_match_score=subset_match_score)
        
#         rouge_results = self.rouge_metric.compute() 
#         rouge_dict = self.parse_score(rouge_results)    
        return base_metrics
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
  
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)
    
  
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        
        if len(self.em_score_list) <= 2:
            average_em_score = sum(self.em_score_list) / len(self.em_score_list) 
            average_subset_match_score = sum(self.subset_score_list)/len(self.subset_score_list)
            
        else:
            latest_em_score = self.em_score_list[:-2]
            latest_subset_score = self.subset_score_list[:-2]
            average_em_score = sum(latest_em_score) / len(latest_em_score) 
            average_subset_match_score = sum(latest_subset_score)/len(latest_subset_score)
            
        
        
        average_em_score = torch.tensor(average_em_score,dtype=torch.float32)
        average_subset_match_score = torch.tensor(average_subset_match_score,dtype=torch.float32)
        tensorboard_logs.update(em_score=average_em_score, subset_match_score=average_subset_match_score)
        
        ## Clear out the lists for next epoch
        self.target_gen= []
        self.prediction_gen=[]
        return {"avg_val_loss": avg_loss, 
                "em_score" : average_em_score,
                "subset_match_score" : average_subset_match_score,
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False,
                             relative_step=False)
        self.opt = optimizer
        return [optimizer]
  
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=False):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()
  
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict
    

    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        sampler=RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=0)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        t_total = 100000
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        
        validation_dataset = get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)
        sampler=RandomSampler(validation_dataset)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, sampler =sampler, num_workers=0)
    
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=0)
    
    
    def on_save_checkpoint(self, checkpoint):
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

prefix_path='' #Path to custom training data. Name the training corpus train_context.csv
class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
      self.dataset = self.split_into_segment(pd.read_csv(prefix_path+"train_context.csv"),input_length)
      self.input_length = input_length
      self.tokenizer = tokenizer
      self.output_length = output_length
      self.print_text = print_text
  
    def split_into_segment(self, ds, input_length):
        new_rows = []
        for index, row in ds.iterrows():
            if len(row['context'].split()) > input_length:
                word_list = row['context'].split()
                seg1 = word_list[:input_length]
                segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
                segment2 = seg2_a + (' '.join(word_list[input_length:]))
                ds.loc[index, 'context'] = segment1
                while(len(segment2.split()) > input_length):
                    word_list = segment2.split()
                    seg1_ = word_list[:input_length]
                    if '.' in ' '.join(seg1_):
                        segment1_, seg2_a_ = (' '.join(seg1_)).rsplit('.',1)
                        segment2 = seg2_a_ + (' '.join(word_list[input_length:]))
                    else:
                        segment1_ = ' '.join(seg1_)
                        segment2 = (' '.join(word_list[input_length:]))
                    new_rows.append(segment1_)
                new_rows.append(segment2)
        ds2 = pd.DataFrame(new_rows, columns=['context'])
        ds = ds.append(ds2)
        return ds

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        
        return text

    def span_corruption_mask(self, text, noise_span_length=3, noise_density=.15):
        max_index = len(text.split())
        mask = max_index * [0]
        span_num = math.ceil(( max_index * noise_density ) / 3 )
        exclude=[max_index-2, max_index-1]
        for i in range(span_num):
            while True:
                rand_num = np.random.randint(low=0, high=max_index) #Getting random number for mask index
                if rand_num not in exclude:
                    span = [rand_num, rand_num+1, rand_num+2]
                    for s in span:
                        mask[s] = 1
                        exclude.append(s)
                    if rand_num==1:
                        exclude.append(rand_num-1)
                    elif rand_num==2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                    elif rand_num>2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                        exclude.append(rand_num-3)
                    if not rand_num==max_index-3:
                        exclude.append(span[-1]+1)
                    break
                else:
                    continue
        return mask
    
    def noise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        one_count=0
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 1:
                one_count+=1
                if one_count==1:
                    text_.append(sentinels[sentinel_cnt])
                    sentinel_cnt+=1
                else:
                    if one_count==3:
                        one_count=0
            else:
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def nonnoise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        zero_first=True
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 0:
                if zero_first:
                    text_.append(sentinels[sentinel_cnt])
                    zero_first=False
                    sentinel_cnt+=1
            else:
                zero_first=True
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['context']))
        text = self.clean_text(example_batch['context'])
        mask = self.span_corruption_mask(text)
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        input_ = self.noise_span_to_unique_sentinel(text,mask,sentinels)
        target_ = self.nonnoise_span_to_unique_sentinel(text,mask,sentinels)
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

def get_dataset(tokenizer, type_path, num_samples, args):
    return Pretrain(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                        output_length=args.max_output_length)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_length', default=128)
    parser.add_argument('--output_length', default=128)
    parser.add_argument('--num_train_epochs', default=1)
    parser.add_argument('--output_dir', default='t5_pretraining')
    parser.add_argument('--train_batch_size', default=8)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--model', default='t5-base')
    hparam = parser.parse_args()

    args_dict = dict(
        output_dir="", # path to save the checkpoints
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,
        max_input_length=int(hparam.input_length),
        max_output_length=int(hparam.output_length),
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=1e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=4,
        eval_batch_size=4,
        num_train_epochs=2,
        gradient_accumulation_steps=1,
        n_gpu=1,
        resume_from_checkpoint=None, 
        val_check_interval = 1.0,
        n_val=0,
        val_percent_check= 0,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=101,
    )

    args_dict.update({'output_dir': hparam.output_dir, 'num_train_epochs':int(hparam.num_train_epochs),
                    'train_batch_size': int(hparam.train_batch_size), 'eval_batch_size': int(hparam.train_batch_size), 'learning_rate': float(hparam.learning_rate)})
    args = argparse.Namespace(**args_dict)

    ## Define Checkpoint function
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint")

    ## If resuming from checkpoint, add an arg resume_from_checkpoint
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
        callbacks=[LoggingCallback()]
    )
    
    set_seed(42)
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)