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
    parser.add_argument("--batch_num", type=int, default=0, help="Batch number to evaluate.")
    parser.add_argument("--max-length", type=int, default=16, help="Soft upper bound for the number of dialog turns (in total) shown to the participant.")
    parser.add_argument("--instructions", type=str, default="instructions.md", help="Path to the instructions markdown file.")
    parser.add_argument("--checks", type=str, help="Path to the sanity checks json file.")
    args = parser.parse_args()
    return args


@st.cache_data
def load_data_MINE(inputs):
    with open(inputs) as f:
        generated = json.load(f)
    id2gen = {dial_num: datapoint[dial_num]  for datapoint in generated for dial_num in datapoint}
    print(f"{len(id2gen)} datapoins loadded.")
    id_list = list(id2gen.keys())
    return id2gen, id_list


def prepare_data(datapoints):
    if 'id_list' in st.session_state and \
       'id2gen' in st.session_state:  
       return

    with st.spinner('Loading data, please wait ...'):
        id2gen, id_list = load_data_MINE(datapoints)
        
    st.session_state['id_list'] = id_list
    st.session_state['id2gen'] = id2gen
    st.session_state['example_clicked'] = False
    st.session_state['metrics'] = [ 
                                "Situation-groundedness",
                                "Supportiveness",
                                "Naturalness",
                                ]


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

        # .stRadio > div > label:first-of-type {
        #     display: none
        # }
        
        .stRadio > label {
            float: left
        }
        
        .stRadio > div {
            position: relative;
            float: left;
            flex-direction: row;
            justify-content: space-around;
            width: 800px;
        }

        .stRadio > div:after {
        #     content: '←  Best';
            position: absolute;
            left: 100%;
            top: 1px;
            width: 75px;
            font-size: 0.9rem
        }

        # .stRadio > div > label {
        #     background: rgb(150, 150, 190)
        # }

        .stRadio > div > label:hover {
            background: rgb(150, 150, 150)
        }

        .stRadio > div > label > div:first-of-type {
            position: relative;
            left: 25px;
        }
        
        .stRadio > div > label > div:last-of-type {
            position: relative;
            left: -25px;
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
            participant_id = "participant_" + name + "_" + str(args.batch_num)
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
            dialog_id: {name:0 for name in st.session_state["metrics"]} for dialog_id in st.session_state['id_list']}
        # print(st.session_state['ratings'])        

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
            

            dial_num = st.session_state['id_list'][ex_idx]
            situation = st.session_state['id2gen'][dial_num]['situation']
            history = st.session_state['id2gen'][dial_num]['context']
            usr_utt = st.session_state['id2gen'][dial_num]['usr_utt']
            utt_with_backstory = st.session_state['id2gen'][dial_num]['utt_with_backstory']
            sys_resp = st.session_state['id2gen'][dial_num]['sys_resp']
            resp_with_reaction = st.session_state['id2gen'][dial_num]['resp_with_reaction']

            st.markdown(f"- **User's Situation**")
            st.markdown(situation.strip())


            st.markdown(f"- **Previous turns**")
            if history == "None\n":
                st.markdown("None")
            else:
                history = history.split(',\n')
                for idx, turn in enumerate(history):
                    turn = turn.replace("User: ", "").replace("System: ", "")[1:-1]                  
                    if idx % 2 == 0:
                        st.warning(f"*{idx + 1}. USR:* {turn}")
                    else:
                        st.info(f"*{idx + 1}. SYS:* {turn}")

            st.markdown(f"- :sparkles:**Exchange to be rated** :sparkles:")
            # make user utterance
            pattern = r'\*\*(.*?)\*\*'
            og_utt = re.findall(pattern, utt_with_backstory)[0]
            if og_utt[-1] not in ['.', '!', '?']:
                og_utt += '.'
            pattern = r'<Backstory:\s*(.*?)\s*>'
            try :
                backstory = re.findall(pattern, utt_with_backstory)[0]
            except:
                backstory = ""
            final_usr = og_utt + " " + backstory

            # make system utterance
            pattern = r'\*\*(.*?)\*\*'
            og_resp = re.findall(pattern, resp_with_reaction)[0]
            pattern = r'<Reaction:\s*(.*?)\s*>'
            reaction = re.findall(pattern, resp_with_reaction)[0]
            final_sys = reaction + " " + og_resp
    
            st.info(f"**USER UTTERANCE:** {final_usr}")
            st.warning(f"**SYSTEM UTTERANCE:** {final_sys}")

           
            
            st.write("#### 🏆 &nbsp; Your ratings")
             #st.write("Please select an option for each rating before submitting your ratings.")    

            def next_example(): 
                # st.write("**Ratings submitted! :tada:**")
                st.session_state['valid_rating'] = False
                for metric in st.session_state['metrics']:
                    score = getattr(st.session_state, f"rating_example{dial_num}_{metric}")
                    try :
                        score = int(score)
                    except TypeError:
                        break
                    st.session_state['ratings'][dial_num][metric] = score
                    st.session_state['valid_rating'] = True
                   
                
                # pp.pprint(st.session_state['ratings'])
            
            with st.form(key='my_form'):

                for metric in st.session_state['metrics']:
                    # with cols[permutation[i]]:
                    c = st.container()
                    if metric == "Situation-groundedness":
                        c.markdown(f"**In the user utterance, assess whether the backstory given by the user can indeed be derived from \
                                   their situation.**")
                        c.radio(" ", [str(x) for x in range(1,4)], index=None, key=f'rating_example{dial_num}_{metric}',
                                help="""- **Not at all**: the backstory connot be derived from the initial situation. 
                                        - **Somewhat**: the backstory con only be partly derived from the initial situation. 
                                        - **Fully**: the backstory can be dervied from the initial \
                                    situation.""",
                                captions=["Not at all", "Somewhat", "Fully",])
                    
                    elif metric == "Supportiveness":
                        c.markdown(f"**For the system response, assess whether it provides support and understanding with respect to \
                                   the user's backstory.**")
                        c.radio(" ", [str(x) for x in range(1,4)], index=None, key=f'rating_example{dial_num}_{metric}',
                                help="""- **Not at all**: the response does not display support or understanding. 
                                        - **Somewhat**: the response shows some support and understanding. 
                                        - **Fully**: the response shows appropriate support and understanding.""",
                                captions=["Not at all", "Somewhat", "Fully",])
                    # elif metric == "Specificity":
                    #     c.markdown(f"**Is the system utterance specific to the user's utterance?**")
                    #     c.radio(" ", [str(x) for x in range(1,5)], index=None, key=f'rating_example{dial_num}_{metric}',
                    #             help="""- **Generic**: the system's response could be used in many different situations.
                    #                     - **Specific enough**: the system's response is specific to the user's \
                    #                 utterance, but could potentially be used in other situations as well.
                    #                     - **Highly specific**: the system's response is highly specific to the user's \
                    #                 utterance.""",
                    #             captions=["No answer", "Generic", "Specific", "Highly specific"])
                    
                    elif metric == "Naturalness": 
                        c.markdown(f"**Overall, does the exchange sound natural and coherent?**")
                        c.radio(" ", [str(x) for x in range(1,4)], index=None, key=f'rating_example{dial_num}_{metric}',
                                help="""- **Not at all**: the exchange sounds unnatural and is hard to follow.
                                        - **Somewhat**: the exchange sounds natural enough.
                                        - **Fully**: the exchange sounds completely fluent.""", 
                                captions=["Not at all", "Somewhat", "Fully"])
                    
                    
                    # elif metric == "Factuality":
                    #     c.markdown(f"Look at the segment below, does it display any **specific factual knowledge** (information \
                    #                 about museum exhibits, restaurant menus, locations...), **general knowledge** \
                    #                or **none of the above** ?")
                    #     # Binary radio buttons with choices Yes/No
                    #     c.markdown(f"The segment to consider is: \"*{extract_for_check}*\"")
                    #     c.radio(" ", [str(x) for x in range(1,4)], index=0, key=f'rating_example{dial_num}_{metric}',
                    #             captions=["None", "Specific Knowledge", "General Knowledge", ])

               
                if st.form_submit_button("Submit Ratings", on_click=next_example):
                    print(st.session_state['valid_rating'])
                    if not st.session_state['valid_rating']:
                        st.write("**Please fill in all ratings before submitting!**")
                    else:
                        st.write("**Ratings submitted! :tada:**")
                    
                        

            break
    else:
        st.success("Please select an example to rate.")
        

                
if __name__ == '__main__':

    initialize()
    args = parse_args()
    datapoints = f"./batches/batch_{args.batch_num}.json"
    prepare_data(datapoints)
    run_application(args)
