import os
import json
import copy
import argparse

def prep_simpletod(delex, fusedchat, variant='simpletod'):
    """format examples for different variants"""
    datapoint_dict = {}
    for d_num in delex:
        dial = delex[d_num]
        augmented_idx = dial["augmented_idx"]
        context = "<|context|> "
        datapoint_dict[d_num] = []

        if variant == 'fusedchat':
            rewrite = False
            fchat_turns = fusedchat[d_num]["turns"]
            fchat_types = fusedchat[d_num]["types"]
            for idx, (typ, turn) in enumerate(zip(fchat_types, fchat_turns)):
                string_encode = turn.encode("ascii", "ignore") # get rid of unicode characters in fusedchat
                turn = string_encode.decode()
                if typ == 'prepended':
                    if idx % 2 == 0:
                        context += f"<|user|> {turn} "
                    else:
                        example_context = context + "<|endofcontext|> "
                        response = f"<|response|> {turn} <|endofresponse|>"
                        datapoint_dict[d_num] += [{"input_context": example_context, "output": response}]
                        context += f"<|system|> {turn} "
                    
                elif typ == "rewritten":
                    # only occur on user turns
                    context += f"<|user|> {turn} "
                    rewrite = True
                    break

                elif typ == 'original':
                    break

        for idx, turn in enumerate(dial['log']):
            if variant == 'fusedchat' and rewrite == True and idx == 0:
                continue

            if idx%2 == 0: 
                turn['text'] = ' '.join(turn['text'].split()) # one line per example => eliminate any newlines
                context += f"<|user|> {turn['text']} " # user will always be lexicalized.
                if variant == 'interference' and idx == augmented_idx[0]:
                    context += turn["backstory"] + " "

            else:
                # context
                example_context = context + "<|endofcontext|> "
                
                # belief
                belief = '<|belief|> '
                for domain in turn['metadata']:
                    constraint = turn['metadata'][domain]
                    for b in constraint['book']:
                        if b != 'booked' and constraint['book'][b] != []:
                            belief += f"{domain} book {b.lower()} {constraint['book'][b][0].lower()}, " # if multiple values are considered correct, we pick the first one
                            
                    for b in constraint['semi']:
                        if constraint['semi'][b] != 'not mentioned' and constraint['semi'][b] != []: 
                            belief += f"{domain} {b.lower()} {constraint['semi'][b][0].lower()}, " 
                belief = belief[:-2] + ' ' if belief[-2] == ',' else belief # remove last comma
                belief += '<|endofbelief|> '

                # action
                action = '<|action|> '
                turn_acts = turn['dialog_act']
                name_acts = [] 
                other_acts = []
                for act in turn_acts:        
                    for slot, _ in turn_acts[act]:
                        act = act.replace('-', ' ').lower() # Hotel-Inform => hotel inform
                        if slot == 'none':
                            other_acts.append(act)
                        elif slot.lower() == "name":
                            name_acts.append(f"{act} {slot.lower()}")
                        else:
                            other_acts.append(f"{act} {slot.lower()}")
                list_acts = name_acts + other_acts
                # if variant == 'interference' and idx == augmented_idx[1]:
                #     list_acts = ['supportive_reaction'] + name_acts + other_acts
                action += ', '.join(list_acts)
                action += ' <|endofaction|> '
                

                # response
                turn['delex_text'] = ' '.join(turn['delex_text'].split()) 
                if variant == 'interference' and idx == augmented_idx[1]:
                    reaction = turn["reaction"]
                    response = f"<|response|> {reaction} {turn['delex_text']} <|endofresponse|>"
                else:
                    response = f"<|response|> {turn['delex_text']} <|endofresponse|>"
                    
                # add example
                datapoint_dict[d_num] += [{"input_context": example_context, "output": belief + action + response}]
            
                # add lexicalized resp to history
                turn['text'] = ' '.join(turn['text'].split())
                if variant == 'interference' and idx == augmented_idx[1]:
                    context += f"<|system|> {reaction}" + ' ' + f"{turn['text']} "
                else :
                    context += f"<|system|> {turn['text']} " 
    
    return datapoint_dict

def delexicalize(dialogues):
    delex = copy.deepcopy(dialogues)
    for dial_num in delex:
        for idx, turn in enumerate(delex[dial_num]["log"]):
            # delexicalize using span info
            if idx % 2 == 0:
                continue
            text = turn["text"]
            delex_spans = turn["span_info"]
            char_diff = 0
            for span in delex_spans:
                act, slot, value, start, end = span
                start += char_diff
                end += char_diff
                len1 = len(text)
                text = text[:start] + f'[{slot}]' + text[end:]
                len2 = len(text)
                char_diff += len2 - len1
            turn["delex_text"] = text
    return delex
    


if __name__ == '__main__':

    with open('interference_data/test.json') as f:
        test = json.load(f)
    
    with open('interference_data/train.json') as f:
        train = json.load(f)

    with open('interference_data/valid.json') as f:
        valid = json.load(f)

    with open('fusedchat_prepended.json') as f:
        fusedchat = json.load(f)

        
    save_dir ='lm_data/'
    txt_data_dir = os.path.join(save_dir, 'txt_data/')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(txt_data_dir, exist_ok=True)

    for var in ['simpletod', 'interference', 'fusedchat']:
        os.makedirs(os.path.join(txt_data_dir, var), exist_ok=True)
        output_dict = {}
        for split, split_name in zip([train, valid, test], ['train', 'valid', 'test']):
            delex = delexicalize(split)
            print(f"{split_name} size: {len(delex)}")
            processed = prep_simpletod(delex, fusedchat, variant=var)
            output_dict[split_name] = processed 
            # write as txt file to pass to language modeling
            with open(os.path.join(txt_data_dir, var, f'{split_name}.txt'), 'w') as f:
                for dial_num in processed:
                    for turn in processed[dial_num]:
                        f.write(turn["input_context"] + turn["output"] + '\n')

        # save full variant as dict
        with open(os.path.join(save_dir, f'{var}.json'), 'w') as f:
            json.dump(output_dict, f, indent=2)   