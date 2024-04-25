# Chitchat as Interference: Adding User Backstories to Task-Oriented Dialogues
During task-oriented dialogues (TODs), human users naturally introduce questions or information that is beyond the immediate scope of the task. This creates interferences which TOD systems cannot always handle. In this project, we create a testbed for building more resilient TOD systems. Using few-shot prompting with an LLM, we create a novel chitchat augmentation for MultiWOZ. In this augmentation a chatty user adds elements of backstory to their request, an interference to which the system responds with support and understanding while also advancing the task. We assess the resilience of 3 baselines, all based on SimpleToD. 

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

## Generating the Interferences
Augmented dialogues can be directly found in `data/interference_data/`. Augmented dialogues will have an `augmented_idx` key with the idxs of the turns augmented.  For the augmented turns, the user will have a `backstory` key with the backstory to append and the system will have a `reaction` key with the supportive reaction to prepend to the original text.  

Here is the pipeline used for reference or modification:

1. Download the MultiWOZ2.2 dataset from [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2). Follow guidelines to convert the data to MultiWOZ2.1 format at the bottom of the page: you should have one single `.json` with all the annotated dialogues. Also download Fusedchat dialogues from [here](https://github.com/tomyoung903/FusedChat). The file needed is `fusedchat_prepended.json`.  Add both `.jsons` to `data`.

2. Generate seed situations in a few-shot manner using the prepended FusedChat exchanges.  Check the script for paths are correct and set the desired arguments. You can separate the training data into several bathces to parallelize if needed. (Don't forget to merge the split back together afterwards.)
```
python gen_situation.py --model_name=<...> --data_split=<...> --training_batch_number=<...>
``` 
3. Augment random user turns with backstory and the following system responses with a supportive reaction. Pass in the path for the generated situations. Optiionally add the backstory path if already generated. Check argparse arguments for more details.
```
python gen_chitchat.py --model_name=<...> --gen_sit_path=<...> --gen_back_path=<...>
```
For steps 2. and 3. you can also use and modify the following scripts to run the jobs on slurm:
```
sbatch launch_gen.sh
```
4. Filter and merge interferences into mwoz: 
```
python merge_interferences.py --interference_path=<...> --split=<...>
```  

## Preparing Data for Training
Prepare training data for SimpleToD. In the `data` directory, run `python prep_simpletod_data.py`. Check paths are correct in the script. This will output an `lm_data` directory.

## Train, gen and eval 
Train the LLM (Llama-2-7B, but any other LLM is possible—just be sure to adapt the `LoraConfig`), generate outputs on an eval set and evalute the generated outputs. This can be done all in one go. Check the script to make sure paths are correct and set the desired arguments.
```
python train_and_generate_and_eval.py --train_data_dir=<...> --eval_data_json=<...> --eval_split=<...> --lr=<...> etc.

# you can also modify the following script to run the job on slurm
sbatch  launch_train_and_generate_and_eval.sh
```

Possible train+eval setups are:
- train on vanilla mwoz | eval: vanilla mwoz (for reference)
- train on vanilla mwoz | eval: interference
- train on fusedchat | eval: interference
- train on interference | eval: interference

To perform generation and evaluation on a model already fine-tuned, run the `gen_and_evaluate.py` script or modify and submit `launch_gen_and_eval.sh`.

### Eval Interface
Interfaces for evalauting the quality of the generated interferences and of the different model responses are in the `eval` directory.
