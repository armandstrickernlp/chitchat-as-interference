import json
import os
import pprint as pp
import argparse
import re
from fuzzywuzzy import fuzz

def jaccard(str1, str2):
    react_tokens = set(str1.split())
    resp_tokens = set(str2.split())
    intersection = len(react_tokens.intersection(resp_tokens))
    union = len(react_tokens) + len(resp_tokens) - intersection
    return intersection / union

def check_requestables(sys_react):
    # want to keep generated response if there are entity names as theese can be contextual, but eliminate if requestable slot info is given
    # 'PHONE', 'ADDRESS', 'POST', 'REFERENCE', 'TRAINID'
    phone = r'\d{10,11}'
    address = r' street | avenue | road | drive | lane | boulevard | court | square | parkway | circle | highway | passage '
    post = r'[a-z]{2}\d{2,3}[a-z]{2}'
    reference = r'reference'
    trainid = r'TR\d{4}'
    return re.search(phone, sys_react) or re.search(address, sys_react) or re.search(post, sys_react) or re.search(reference, sys_react) or re.search(trainid, sys_react)
               

def merge_into_mwoz(gen, mwoz):
    
    keep_gens = []
    interference_data = {}
    rejected_list = []

    for idx, dial_num in enumerate(gen):
        mwoz_dial = mwoz[dial_num+'.json']
        rand_idx = gen[dial_num]["rand_idx"]
        mwoz_dial["augmented_idx"] = (rand_idx, rand_idx+1)

        # reject if user utt structure is not correct
        usr = gen[dial_num]["utt_with_backstory"]
        struct = r'\*\*(.*?)\*\*\s*\+\s*<'
        match = re.findall(struct, usr)
        if len(match) == 1:
            usr_backstory = usr.split("Backstory: ")[1].split('>')[0]
        else:
            rejected_list.append(dial_num)
            continue

        # reject if response structure is not correct
        generated_resp = gen[dial_num]["resp_with_reaction"]
        struct = r'>\s*\+\s*\*\*(.*?)\*\*'
        match = re.findall(struct, generated_resp) # should be 1 match
        if len(match) == 1:
            sys_react = generated_resp.split("Reaction: ")[1].split('>')[0]

            # check no requestables are in the reaction
            sys_react_low = sys_react.lower()
            if check_requestables(sys_react_low):
                rejected_list.append(dial_num)
                continue
            
            # check reaction is not too similar to the matched text (og response)
            jacc_simil = jaccard(match[0], sys_react)
            if jacc_simil>0.25:
                rejected_list.append(dial_num)
                continue
            
            ratio = fuzz.ratio(match[0], sys_react)
            if ratio>50:
                rejected_list.append(dial_num)
                continue

            if match[0] in sys_react: 
                # sometimes the reaction mixes up the orignal response and the reaction, which leads to a confusing overall response
                rejected_list.append(dial_num)
                continue
            
            # if no issues, merge into mwoz
            mwoz_dial["log"][rand_idx]["backstory"] = usr_backstory
            mwoz_dial["log"][rand_idx+1]["reaction"] = sys_react

            # and add to interference data
            interference_data[dial_num] = mwoz_dial

            # add to list of generations to keep
            keep_gens.append({dial_num: gen[dial_num]})
        
        else:
            rejected_list.append(dial_num)
            continue
   

    return interference_data, rejected_list, keep_gens

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--interference_path", type=str, help="path to gen_reactions")
    parser.add_argument("--split", type=str, default="valid")
    args = parser.parse_args()

    with open(args.interference_path) as f:
        gen = json.load(f)

    with open("data/MultiWOZ_2.2.json") as f:
        mwoz = json.load(f)
    
    interference_data, rejected_list, clean_gens = merge_into_mwoz(gen, mwoz)

    out_path = "data/interference_data/"
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, f"{args.split}.json"), 'w') as f:
        json.dump(interference_data, f, indent=2)

    # out_path = "outputs/clean/"
    # os.makedirs(out_path, exist_ok=True)
    # with open(os.path.join(out_path, f"{args.split}.json"), 'w') as f:
    #     json.dump(clean_gens, f, indent=2)
    
