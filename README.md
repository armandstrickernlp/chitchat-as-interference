# Chitchat as Interference: Adding User Backstories to Task-Oriented Dialogues
During task-oriented dialogues (TODs), human users naturally introduce questions or information that is beyond the immediate scope of the task. This creates interferences which TOD systems cannot always handle. In this project, we create a testbed for building more resilient TOD systems. Using few-shot prmpting with an LLM, we create a novel chitchat augmentation for MultiWOZ. In this augmentation a chatty user adds elements of backstory to their request, an interference to which the system responds with support and understanding while also advancing the task. We assess the resilience of 3 baselines, all based on SimpleToD. 

## Setup
This project uses Python 3.11

Create a virtual environment:

```
conda create -n cc_as_inter python=3.11
```

Install the requirements:
```
git clone git@github.com:armandstrickernlp/chitchat-as-interference.git
cd chitchat-as-interference
pip install -r requirements.txt
```


### Data 
Download the MultiWOZ2.2 dataset from [here](https://huggingface.co/datasets/multi_woz_v22) or [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2). Follow guidelines to convert the data to MultiWOZ2.1 format: you should have one single `.json` with all the annotated dialogues. Also download Fusedchat dialogues with prepended chitchat from [here](https://github.com/tomyoung903/FusedChat). The file needed is `fusedchat_prepended.json`.  Add them to the `data` directory.

## Generating the Interferences
1. Generate the situations from the prepended FusedChat exchanges.  Check the script to make sure paths are correct and set the desired arguments.
```
python gen_situations.py --model_name=<...> --data_split=<...> --training_batch_number=<...>
``` 
2. Augment user turns with backstory and the system response with a supportive reaction. Check the script to make sure paths are correct and set the desired arguments.
```
python gen_chitchat.py --model_name=<...> --gen_sit_path=<...> --gen_back_path=<...>
```
3. Combine train batches into a single json with `python combine_train_batches.py`.
4. Filter and merge interferences into mwoz: `python merge_interferences.py --interference_path=<...> --split=<...>`.  Dialogues will have an `augmented_idx` key with the idxs of the turns augmented.  At those turns, the user will have a `backstory` key with the backstory to append and the system will have a `reaction` key with the supportive reaction.


## Preparing Data for Training
Prepare training data for SimpleToD. IN the `data` directory, run `python prep_simpletod_data.py`. Check paths are correct in the script. This will output an `lm_data` directory.

## Train, gen and eval 
Train the LLM (Llama-2-7B, but any other LLM is possible—just be sure to adapt the `LoraCOnfig`), generate outputs on an eval set and evalute the generated outputs. This can be done all in one go. Check the script to make sure paths are correct and set the desired arguments.
```
python train_and_generate_and_eval.py --train_data_dir=<...> --eval_data_json=<...> --eval_split=<...> --lr=<...> ....

# you can also modify the bash script to run the job on slurm
sbatch  launch_train_and_generate_and_eval.sh
```

Possible train+eval setups are:
- train on vanilla mwoz | eval: vanilla mwoz (for reference)
- train on vanilla mwoz | eval: interference
- train on fusedchat | eval: interference
- train on interference | eval: interference

To perform generation and evaluation on model already fine-tuned, run the `gen_and_evaluate.py` script or modify and submit `launch_gen_and_eval.sh`.

### Eval Interface
Interfaces for evalauting the quality of the generated interferences and of the different model responses are in the `eval` directory. 
``` 
