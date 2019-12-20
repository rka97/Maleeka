from pomegranate import *
import numpy as np
from pom_helpers import *

CODEBOOK_VECTORS = 64

def get_uni_distribution():
    dictionary = {}
    for i in range(CODEBOOK_VECTORS):
        dictionary[str(i)] = 1.0/CODEBOOK_VECTORS
    return DiscreteDistribution(dictionary)

def one_character_model(num_states, char_name):
    char = str(char_name)

    states = []
    initial_state = State(None, name="init")
    # states.append(initial_state)
    for i in range(num_states - 2):
        states.append(State(DiscreteDistribution({"0": 1}), name="s"+str(i)))
    final_state = State(DiscreteDistribution({char: 1}) , name="end")
    # states.append(final_state)
    model = HiddenMarkovModel(name="CharModel", start=initial_state, end=final_state)

    model.add_states(states)
    model.add_transition(model.start, states[0], 1.0/2.0)
    model.add_transition(model.start, model.end, 1.0/2.0)

    nonterm_states = num_states - 2

    model.add_transition(states[0], states[1], 1.0/2.0)
    model.add_transition(states[0], model.start, 1.0/2.0)

    for i in np.arange(start=1, stop=(nonterm_states-1), step=1):
        model.add_transition(states[i], states[i+1], 1.0/2.0)
        model.add_transition(states[i], states[i], 1.0/2.0)
    
    model.add_transition(states[nonterm_states-1], model.end, 1.0/2.0)
    model.add_transition(states[nonterm_states-1], states[nonterm_states-2], 1.0/2.0)

    model.bake()
    return model

# get_uni_distribution()
model = one_character_model(6, "a")
# model = HiddenMarkovModel( "Global Alignment")

# # Define the distribution for insertions
# i_d = DiscreteDistribution( { 'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25 } )

# # Create the insert states
# i0 = State( i_d, name="I0" )
# i1 = State( i_d, name="I1" )
# i2 = State( i_d, name="I2" )
# i3 = State( i_d, name="I3" )

# # Create the match states
# m1 = State( DiscreteDistribution({ "A": 0.95, 'C': 0.01, 'G': 0.01, 'T': 0.02 }) , name="M1" )
# m2 = State( DiscreteDistribution({ "A": 0.003, 'C': 0.99, 'G': 0.003, 'T': 0.004 }) , name="M2" )
# m3 = State( DiscreteDistribution({ "A": 0.01, 'C': 0.01, 'G': 0.01, 'T': 0.97 }) , name="M3" )

# # Create the delete states
# d1 = State( None, name="D1" )
# d2 = State( None, name="D2" )
# d3 = State( None, name="D3" )

# # Add all the states to the model
# model.add_states( [i0, i1, i2, i3, m1, m2, m3, d1, d2, d3 ] )

# # Create transitions from match states
# model.add_transition( model.start, model.start, 0.9 )
# model.add_transition( model.start, i0, 0.1 )
# model.add_transition( m1, m2, 0.9 )
# model.add_transition( m1, i1, 0.05 )
# model.add_transition( m1, d2, 0.05 )
# model.add_transition( m2, m3, 0.9 )
# model.add_transition( m2, i2, 0.05 )
# model.add_transition( m2, d3, 0.05 )
# model.add_transition( m3, model.end, 0.9 )
# model.add_transition( m3, i3, 0.1 )

# # Create transitions from insert states
# model.add_transition( i0, i0, 0.70 )
# model.add_transition( i0, d1, 0.15 )
# model.add_transition( i0, m1, 0.15 )

# model.add_transition( i1, i1, 0.70 )
# model.add_transition( i1, d2, 0.15 )
# model.add_transition( i1, m2, 0.15 )

# model.add_transition( i2, i2, 0.70 )
# model.add_transition( i2, d3, 0.15 )
# model.add_transition( i2, m3, 0.15 )

# model.add_transition( i3, i3, 0.85 )
# model.add_transition( i3, model.end, 0.15 )

# # Create transitions from delete states
# model.add_transition( d1, d2, 0.15 )
# model.add_transition( d1, i1, 0.15 )
# model.add_transition( d1, m2, 0.70 ) 

# model.add_transition( d2, d3, 0.15 )
# model.add_transition( d2, i2, 0.15 )
# model.add_transition( d2, m3, 0.70 )

# model.add_transition( d3, i3, 0.30 )
# model.add_transition( d3, model.end, 0.70 )

# # Call bake to finalize the structure of the model.
# model.bake(verbose=True)

# # plt.figure( figsize=(20, 16) )
# # model.plot()


show_model(model, figsize=(5, 5), filename="example.png", overwrite=True, show_ends=False)

