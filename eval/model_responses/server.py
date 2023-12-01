import string
import random
import json
import argparse
import time
import os
import pprint as pp
import re

import streamlit as st

# add split number to output path

def initialize():
    st.set_page_config(
        page_title="Dialogue evaluation",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        #menu_items={}
    )

@st.cache_data
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--inputs', type=str)
    parser.add_argument("--batch_num", type=int, default=1, help="Batch number.")
    parser.add_argument("--instructions", type=str, default="instructions.md", help="Path to the instructions markdown file.")
    parser.add_argument("--checks", type=str, help="Path to the sanity checks json file.")
    args = parser.parse_args()
    return args


@st.cache_data
def load_data_MINE(inputs):
    with open(inputs) as f:
        generated = json.load(f)
    id2resps = {dial_num: datapoint[dial_num]  for datapoint in generated for dial_num in datapoint}
    print(f"{len(id2resps)} datapoins loaded.")
    id_list = list(id2resps.keys())
    return id2resps, id_list


def prepare_data(datapoints):
    if 'id_list' in st.session_state and \
       'id2resps' in st.session_state:  
       return

    with st.spinner('Loading data, please wait ...'):
        id2resps, id_list = load_data_MINE(datapoints)
        
    st.session_state['id_list'] = id_list
    st.session_state['id2resps'] = id2resps
    st.session_state['system_names'] = ["simpletod", "fusedchat", "interference"]
    st.session_state['example_clicked'] = False

def run_application(args):

    css = """

        body, div {
            font-family: sans-serif !important;
        }

        ul > li {
            font-size: 1.0rem !important;
            font-weight: 100;
        }

        .stRadio {
            margin-bottom: -20px
        }

        .stRadio > div > label:first-of-type {
            display: none
        }
        
        .stRadio > label {
            float: left
        }
        
        .stRadio > div {
            position: relative;
            float: left;
            flex-direction: row;
            justify-content: space-around;
            width: 400px;
        }

        .stRadio > div:after {
            content: '←  Worst';
            position: absolute;
            left: 100%;
            top: 1px;
            width: 75px;
            font-size: 0.9rem
        }

        .stRadio > div > label {
            background: rgb(200, 200, 200)
        }

        .stRadio > div > label:hover {
            background: rgb(150, 150, 150)
        }

        .stRadio > div > label > div:first-of-type {
            position: relative;
            left: 25px;
        }
        
        .stRadio > div > label > div:last-of-type {
            position: relative;
            left: -28px;
        }

        [data-testid="stSidebar"] .stButton {
            width: 100% !important;
        }

        .stButton > button {
            height: 50px;
            padding: 0 20px;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
        }

        [data-testid="stSidebar"] > div:first-of-type {
            padding-top: 1.5rem;
        }

        .main > div {
            padding-top: 1.0rem;
        }

        .stProgress div {
            height: 0.3rem
        }

        .stAlert > [data-baseweb="notification"] {
            padding: 0.25rem 0.45rem;
            margin-bottom: -0.4rem;
        }

        #your-response-ranking {
            margin-top: 1.25rem;
        }
    
        #past-dialog-utterances {
            margin-top: 0.0rem;
        }

        [data-testid="stForm"] > div > [data-testid="stBlock"] {
            border: solid 1px rgb(70,70,70);
            border-radius: 5px;
            padding: 0.4rem 0.6rem 0.2rem 0.6rem;
        }

        .main [data-testid="stForm"] > div {
            width: 100% !important;
        }

        .main [data-testid="stForm"] > div > .element-container {
            width: 180px !important;
            display: inline-block;
            margin-bottom: 0;
        }

        .main [data-testid="stForm"] > div > .element-container div {
            width: 100% !important;
            display: inline-block
        }

        .main [data-testid="stImage"] {
            margin: 1rem auto;
            max-width: 900px;
        }

        .element-container [data-testid="stImage"] {
            margin: 0;
            left: -80px;
            position: relative;
            top: 35px; 
        }

        #MainMenu {visibility: visible;}
        #footer {visibility: hidden;}
    """
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)
     
    st.sidebar.title('Example List')

    if 'introduced' not in st.session_state or not st.session_state['introduced']:
        def set_introduced():
            st.session_state['introduced'] = True

        st.sidebar.markdown("Welcome! :sparkles: Please read the instructions before you start.")

        def show_markdown(md):
            st.markdown(md)

        with open(args.instructions, 'r') as f:
            line_buffer = f.readlines()
            show_markdown(''.join(line_buffer))
        
        st.button("I understand the task, let's start!", on_click=set_introduced)
        st.stop()
    
    
    button_placeholders = [st.sidebar.empty() for i in range(len(st.session_state['id_list']))]
    
    with st.sidebar.form(key='my-form'):
        name = st.text_area("Please enter your **name** before submitting the survey.", key="participant_name")
        warning_placeholder = st.empty()
        final_submit = st.form_submit_button('Submit the survey')
    

    submitted = False

    if final_submit:
        submitted = True

        completed = True
        for ex_idx in st.session_state['id_list']:
            ratings = list(st.session_state['ratings'][ex_idx].values())
            if sum(ratings) == 0: # all ratings are initialized with 0
                completed = False
                break
        
        if submitted and completed and not st.session_state["participant_name"]:
            warning_placeholder.error("Please enter your name before submitting the survey!")
            completed=False
        
        elif submitted and completed and st.session_state["participant_name"]:
            warning_placeholder.success("Thank you for your participation! :tada:")
        
        elif submitted and not completed: 
            warning_placeholder.error("You have to fill in the study first!") 

        if completed :
            
            st.title("Thank you!")
            st.balloons()

            # participant_id = 'participant_' + ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(6))
            participant_id = "participant_" + name + str(args.batch_num)
            output_directory = f'./human_test_evals/'
            output_filename = participant_id + '.json'

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            with open(os.path.join(output_directory, output_filename), 'w+') as f:
                json.dump({
                    "dialog_ids" : st.session_state['id_list'],
                    "ratings" : st.session_state['ratings'],
                    # "sanity_checks" : st.session_state['sanity_checks']
                }, f, indent=2)

    
    # Conversation page

    if 'ratings' not in st.session_state:
        # initialize ratings
        st.session_state['ratings'] = {
            dialog_id: {name:0 for name in st.session_state["system_names"]} for dialog_id in st.session_state['id_list']}
        # print(st.session_state['ratings'])     

    if 'turn_shuffler' not in st.session_state:
        def get_permutation():
            p = st.session_state['system_names'].copy()
            random.shuffle(p)
            return p

        st.session_state['turn_shuffler'] = {
            dialog_id : get_permutation() for dialog_id in st.session_state['id_list']
        }   

    def select_example():
        st.session_state['example_selected'] = [getattr(st.session_state, f"selection_{id}") for id in st.session_state['id_list']]
        st.session_state['example_clicked'] = True

    for i, id in enumerate(st.session_state['id_list']):
        # print(st.session_state['ratings'])
        if sum(st.session_state['ratings'][id].values()) >= 3:
            button_placeholders[i].button(f'Example {i+1}', key=f"selection_{id}", on_click=select_example, disabled=True)
        else:
            button_placeholders[i].button(f'Example {i+1}', key=f"selection_{id}", on_click=select_example)
    
    st.sidebar.markdown("Please select an example to rate.")


    

    # Example page
    if st.session_state['example_clicked']:

        for ex_idx, marker in enumerate(st.session_state['example_selected']):
            if marker == False:
                continue

            st.write(f"#### 🗨️ &nbsp; Example {ex_idx+1}")
            
            st.markdown(f"- **Previous turns**")
            dial_num = st.session_state['id_list'][ex_idx]
            history = st.session_state['id2resps'][dial_num]['context']
            history = history.replace("<|user|>", '----').replace("<|system|>", '----').replace("<|context|>", '').replace("<|endofcontext|>", '').strip()
            history = history[4:].split('----') # skip '----' at start

            for idx, turn in enumerate(history[-11:]): # show last 10 turns
            
                if idx % 2 == 0:
                    st.info(f"*{idx + 1}. USR:* {turn}")
                else:
                    st.warning(f"*{idx + 1}. SYS:* {turn}")

            
            st.write("#### 🏆 &nbsp; Your response ranking:")    

            def next_example(): 
                for name in st.session_state['system_names']:
                    rating = getattr(st.session_state, f"rating_example{dial_num}_{name}")
                    if rating != "0.":
                        st.session_state['ratings'][dial_num][name] = int(rating)
                print(st.session_state['ratings'][dial_num])

            with st.form(key='my_form'):

                permutation = st.session_state['turn_shuffler'][dial_num]
                num_systems = len(permutation)

                for name in permutation:
                    # with cols[permutation[i]]:
                    c = st.container()
                    resp = st.session_state['id2resps'][dial_num][name]
                    # response = random.choice(args.sanity_sentence) if st.session_state['sanity_checks'][conversation_id][n][turn_idx] else resp

                    # selected_idx = st.session_state['ratings'][ex_idx][name]
                    
                    c.markdown(f"**{resp}**")
                    
                    c.radio("Best → ", [str(x) for x in range(num_systems+1)], index=None, key=f'rating_example{dial_num}_{name}')
                    

                if st.form_submit_button("Submit Ranking", on_click=next_example):
                    st.write("**Response submitted! :tada:**")
           

            break
    else:
        st.success("Please select an example to rate.")
        

                
if __name__ == '__main__':

    initialize()
    args = parse_args()
    if args.checks:
        datapoints = args.checks
    else:   
        datapoints = f"./batches/batch{args.batch_num}.json"
    prepare_data(datapoints)
    run_application(args)
