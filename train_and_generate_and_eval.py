from functools import partial
import argparse
import os
import json
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, 
                          logging,
                          set_seed
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model

import evaluate
import torch

from mwzeval.metrics import Evaluator
from tod_eval_utils import get_predictions_and_JGA


def load_model(model_name):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="./llama2_cache",
        torch_dtype=torch.bfloat16,
        device_map="auto", # dispatch efficiently the model on the available ressources
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./llama2_cache")

    # Needed for tokenizer + special case of our dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', '[address]', '[area]','[arriveby]','[bookday]','[bookpeople]','[bookstay]','[booktime]', '[choice]','[day]','[department]','[departure]','[destination]','[duration]','[entrancefee]','[food]','[leaveat]','[name]','[openhours]','[phone]','[postcode]','[price]','[pricerange]','[ref]','[stars]','[trainid]','[type]']})

    # adapt embedding layer to the new vocabulary
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch : sentencepiece adds <s> to each start of example
    Padding is done dynamically with the collator 
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(dataset, tokenizer, max_length, seed):

    print("Preprocessing dataset...")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["text"],
    )
    # keep examples that have less than max_length tokens
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    # dataset = dataset.shuffle(seed=seed)
    return dataset


if __name__ == "__main__":

    """python train_and_generate_and_eval.py --train_data_dir=data/lm_data/txt_data/simpletod --eval_data_json=data/lm_data/simpletod.json --eval_split=valid --seed=42"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='data/lm_data/txt_data/fusedchat', help='path to training data directory')
    parser.add_argument('--eval_data_json', type=str, default='data/lm_data/interference.json', help='path to json file with data to evaluate on')
    parser.add_argument('--eval_split', type=str, default='valid', help='split to use: valid or test')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    args = parser.parse_args()

    set_seed(args.seed)
    model_name=args.model_name

    os.environ["WANDB_DISABLED"] = "offline"

    base_model, tokenizer = load_model(model_name)

    lora_config = LoraConfig(
        r=args.rank, # matrix dim
        lora_alpha=32, # alpha scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=model_name,
        modules_to_save=['lm_head', 'embed_tokens'],
    )

    peft_model = get_peft_model(base_model, lora_config)

    # # check embeddings and lm_head are trainable
    # for name, param in peft_model.named_parameters():
    #     if 'embed_tokens' in name or 'lm_head' in name:
    #         print(name, param.requires_grad)

    # load dataset
    dataset = load_dataset('text', 
                data_dir=args.train_data_dir,
                split=['train[:]', 'validation[:]']
                )
    dataset = DatasetDict({'train': dataset[0], 'validation':dataset[1]})
    max_length = peft_model.config.max_position_embeddings
    dataset = preprocess_dataset(dataset, tokenizer, max_length, seed=args.seed)


    # training
    training_set = args.train_data_dir.split('/')[-1]
    output_dir = f"training_outputs/{training_set}/{args.lr}_{args.seed}_rank{args.rank}/"
    os.makedirs(output_dir, exist_ok=True)

    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()

    training_arguments = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, # effective batch size of 32
        
        learning_rate=args.lr,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        lr_scheduler_type="linear",    

        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        #############################
        # max_steps=10,
        #############################
        num_train_epochs=args.epochs,

        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        metric_for_best_model='eval_loss',

        #report_to='wandb',
        output_dir=output_dir,
    )

    trainer = Trainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_arguments,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    peft_model.config.use_cache = False

    trainer.train()
    
    # inference
    model = trainer.model
    model.config.use_cache = True
    model.merge_and_unload(progressbar=True)
    model.eval()

    tokenizer.padding_side = "left"
    eos_token = "<|endofresponse|>"

    eval_set = args.eval_data_json.split('/')[-1].split('.')[0]
    output_dir = f"gen_outputs/{training_set}_to_{eval_set}_{args.eval_split}/{args.lr}_{args.seed}_rank{args.rank}/"
    os.makedirs(output_dir, exist_ok=True)

    # load gold data
    with open(args.eval_data_json) as f:
        gold = json.load(f)
        
    generated = {}

    eval_exs = gold[args.eval_split]

    for idx, dial_num in tqdm(enumerate(eval_exs)):
        #############################
        # if idx == 1:
        #     break
        #############################
        dial = eval_exs[dial_num]
        full = []
        batch = []
        batch_gold = []
        for turn in dial:
            batch.append(turn['input_context'])
            batch_gold.append(turn['output'])

        encoding = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, 
                max_length=1500, 
                eos_token_id=tokenizer.convert_tokens_to_ids([eos_token])[0],
                no_repeat_ngram_size=10,
                )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
        for ex, gold, gen in zip(batch, batch_gold, generated_texts):
            gen = gen.replace('</s>', '')
            gen = gen.split('<|endofcontext|>')[1]
            response = gen.split('<|endofresponse|>')[0].split('<|response|>')[-1]
            gold_resp = gold.split('<|endofresponse|>')[0].split('<|response|>')[-1]
            full.append({'input': ex, 'gold': gold, 'generated': gen, 'gold_resp': gold_resp, 'response': response})

        generated[dial_num.replace('.json', '').lower()] = full
        
    with open(os.path.join(output_dir, 'gen.json'), 'w') as f:
        json.dump(generated, f, indent=2)
    

# compute results
results_output_dir = f"gen_results/{training_set}_to_{eval_set}_{args.eval_split}/{args.lr}_{args.seed}_rank{args.rank}/"
os.makedirs(results_output_dir, exist_ok=True)

predictions, JGA = get_predictions_and_JGA(generated, results_output_dir)

e = Evaluator(success=True, richness=True, bleu=False)
res = e.evaluate(predictions)

my_results = {
    "inform" : res["success"]["inform"]["total"],
    "success": res["success"]["success"]["total"],
    "CBE": res["richness"]["cond_entropy"],
    "unique_trigrams": res["richness"]["num_trigrams"],
}

my_results['JGA'] = JGA


# calculate BLEU
bleu = evaluate.load("bleu", cache_dir="./evaluate_cache")

if eval_set == 'simpletod':
    gold, preds = [], []
    for dial_num in generated:
        for idx, turn in enumerate(generated[dial_num]):
            gold.append([turn["gold_resp"].strip()])
            preds.append(turn["response"].strip())
    bleu_score = bleu.compute(predictions=preds, references=gold)
    my_results['BLEU'] = round(bleu_score['bleu'], 3)


elif eval_set == 'interference':
    # bleu scores for aug, orig, all
    gold_aug, gold_orig, gold_all = [], [], []
    pred_aug, pred_orig, pred_all = [], [], []

    with open(f"data/interference_data/{args.eval_split}.json") as f:
        full_dials = json.load(f)
    
    for dial_num in generated:
        aug_idx = full_dials[dial_num.upper()]["augmented_idx"][0] // 2
        for idx, turn in enumerate(generated[dial_num]):

            if idx == aug_idx:
                gold_aug.append([turn["gold_resp"].strip()]) # nested list for gold
                pred_aug.append(turn["response"].strip())
            
            else:
                gold_orig.append([turn["gold_resp"].strip()])
                pred_orig.append(turn["response"].strip())
            
            gold_all.append([turn["gold_resp"].strip()])
            pred_all.append(turn["response"].strip())

    bleu_score_aug = bleu.compute(predictions=pred_aug, references=gold_aug)
    bleu_score_orig = bleu.compute(predictions=pred_orig, references=gold_orig)
    bleu_score_all = bleu.compute(predictions=pred_all, references=gold_all)

    my_results['BLEU_aug'] = round(bleu_score_aug['bleu'], 3)
    my_results['BLEU_orig'] = round(bleu_score_orig['bleu'], 3)
    my_results['BLEU_all'] = round(bleu_score_all['bleu'], 3)


with open(os.path.join(results_output_dir, f'results.json'), 'w') as f: 
    json.dump(my_results, f, indent=2)

   



    



