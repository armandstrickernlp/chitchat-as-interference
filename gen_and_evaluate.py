import argparse
import os
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz

from mwzeval.metrics import Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, logging
from peft import PeftModel
import evaluate
import torch

from tod_eval_utils import get_predictions_and_JGA

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint directory where lora weights are stored')
parser.add_argument("--training_set", type=str, default="simpletod")
parser.add_argument("--eval_data_json", type=str, default="data/lm_data/interference.json")
parser.add_argument("--eval_split", type=str, default="valid")
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--cache_dir', type=str, default='./llama2_cache')

args = parser.parse_args()

set_seed(42)

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

base_model.resize_token_embeddings(len(tokenizer))

peft_model = PeftModel.from_pretrained(base_model, 
                                      args.checkpoint_path,
                                      device_map="auto",
                                      is_training=False,
    )
peft_model = peft_model.merge_and_unload(progressbar=True)
peft_model.eval()

tokenizer.padding_side = "left"
eos_token = "<|endofresponse|>"

# set output path
eval_set = args.eval_data_json.split('/')[-1].split('.')[0]
gen_output_dir = f"gen_outputs/{args.training_set}_to_{eval_set}_{args.eval_split}/{args.checkpoint_path.split('/')[-2]}"
os.makedirs(gen_output_dir, exist_ok=True)

# load gold data
with open(args.eval_data_json) as f:
    gold = json.load(f)
    
generated = {}

eval_exs = gold[args.eval_split]

for idx, dial_num in tqdm(enumerate(eval_exs)):
    #############################
    # if idx == 5:
    #     break
    #############################
    dial = eval_exs[dial_num]
    full = []
    batch = []
    batch_gold = []
    for turn in dial:
        batch.append(turn['input_context'])
        batch_gold.append(turn['output'])

    encoding = tokenizer(batch, return_tensors="pt", padding=True).to(peft_model.device)
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, 
            max_length=1000, 
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
    
with open(os.path.join(gen_output_dir, 'gen.json'), 'w') as f:
    json.dump(generated, f, indent=2)



results_output_dir = f"gen_results/{args.training_set}_to_{eval_set}_{args.eval_split}/{args.checkpoint_path.split('/')[-2]}"
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
bleu = evaluate.load("bleu", cache_dir="evaluate_cache")


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
