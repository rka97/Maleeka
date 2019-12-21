from pomegranate import *
import numpy as np
from pom_helpers import *
from util import *
import pickle

def get_uni_distribution():
    dictionary = {}
    for i in range(NUM_CLUSTERS):
        dictionary[i] = 1.0/NUM_CLUSTERS
    return DiscreteDistribution(dictionary)


def one_character_model(num_states, char_name, char_data):
    char = str(char_name)

    start = State(None, name=char_name + "_start")
    end = State(None, name=char_name + "_end")

    model = HiddenMarkovModel(char_name, start=start, end=end)

    states = []
    for i in range(num_states):
        state = State(get_uni_distribution(), name=char_name + "_s" + str(i))
        states.append(state)
    model.add_states(states)

    model.add_transition(start, states[0], 1)

    for i in np.arange(start=0, stop=num_states-1, step=1):
        model.add_transition(states[i], states[i], 0.5)
        model.add_transition(states[i], states[i+1], 0.5)

    model.add_transition(states[num_states - 1], states[num_states - 1], 0.5)
    model.add_transition(states[num_states - 1], end, 0.5)

    model.bake(merge="None")

    model.fit(sequences=char_data)
    # print(model.states)
    return model, states, start, end


def add_transitions(model, end_state, other_models):
    pr_visit = 1.0 / len(other_models)
    for [_, other_start, _] in other_models:
        model.add_transition(end_state, other_start, pr_visit)


def get_big_model(character_models):
    form_characters = {
        "Beginning": [],
        "End": [],
        "Isolated": [],
        "Middle": []
    }
    
    great_model = HiddenMarkovModel("GreatRecognizer")
    
    for char_name, [model, _, start, end] in character_models.items():
        name, form = char_name.split('_')
        form_characters[form].append( [model, start, end] )
        great_model.add_model(model)
    
    for [model, start, end] in form_characters["Beginning"]:
        add_transitions(great_model, end, form_characters["End"])
        add_transitions(great_model, end, form_characters["Middle"])
    
    for [model, start, end] in form_characters["Isolated"]:
        add_transitions(great_model, end, form_characters["Beginning"])
        add_transitions(great_model, end, form_characters["Isolated"])

    for [model, start, end] in form_characters["Middle"]:
        add_transitions(great_model, end, form_characters["End"])
        add_transitions(great_model, end, form_characters["Middle"])

    for [model, start, end] in form_characters["End"]:
        add_transitions(great_model, end, form_characters["Beginning"])
        add_transitions(great_model, end, form_characters["Isolated"])

    great_model.bake(merge="None")
    show_model(great_model, figsize=(5, 5), filename="example.png", overwrite=True, show_ends=False)

    
    print("Here!")
        

def get_total_model():
    a_model, a_states, a_start, a_end = one_character_model(6, "a")
    b_model, b_states, b_start, b_end = one_character_model(6, "b")

    total_model = HiddenMarkovModel("total")
    total_model.add_model(a_model)
    total_model.add_model(b_model)
    # total_model.add_states([a_start, a_end, b_start, b_end])

    total_model.add_transition(b_end, b_start, 0.5)
    total_model.add_transition(b_end, a_start, 0.5)
    total_model.add_transition(a_end, b_start, 0.5)
    total_model.add_transition(a_end, a_start, 0.5)
    # total_model
    # total_model.add_transition(a_end, b_states[0], 0.3)
    # total_model.add_transition(a_end, b_start, 1.0)
    # total_model.add_transition(a_model.end, a_model.start, 0.5)
    # total_model.add_tusransition(b_model.end, a_model.start, 0.5)
    # total_model.add_transition(b_model.end, b_model.start, 0.5)

    total_model.bake(merge="None")
    print(total_model.states)

    show_model(total_model, figsize=(5, 5), filename="example.png", overwrite=True, show_ends=False)


def train_character_models(letters):
    try:
        with open(LETTER_MODELS_PATH, "rb") as file_handle:
            characters_models = pickle.load(file_handle)
            return characters_models
    except:
        print("Letter samples don't exist, creating them...")
    characters_models = {}
    for letter_name, letter_data in letters.items():
        forms = letter_data["Forms"]
        for form in forms:
            character_name = letter_name + "_" + form
            print("Making HMM for " + character_name)
            character_data = letters[letter_name][form]
            print(len(character_data))
            character_stuff = one_character_model(STATES_PER_CHARACTER, letter_name + "_" + form, character_data)
            characters_models[character_name] = character_stuff
    with open(LETTER_MODELS_PATH, "ab+") as file_handle:
        pickle.dump(characters_models, file_handle)
    return characters_models