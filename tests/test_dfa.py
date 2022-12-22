from constrained_decoding.dfa import DFA

def test_DFA():
    # example from https://stackoverflow.com/questions/35272592/how-are-finite-automata-implemented-in-code/35279645
    dfa_d={ 0:{'0':0, '1':1},
            1:{'0':2, '1':0},
            2:{'0':1, '1':2}}
    dfa = DFA(dfa_d, 0, accept_states={0})
    assert dfa('1011101') == (True, 0, True)
    assert dfa('10111011') == (True, 1, False)
    assert dfa('102') == (False, 2, False)

def test_concat_dfas():
    # example from https://stackoverflow.com/questions/35272592/how-are-finite-automata-implemented-in-code/35279645
    dfa_d={ 0:{'a':1, 'b':2, '0':0},
            1:{'a':2, '0':1},
            2:{'0':2, 'r':0},
            3:{'l':3}}
    dfa1 = DFA(dfa_d, 0, accept_states={2})
    dfa2 = DFA(dfa_d, 2, accept_states={0})
    dfa3 = DFA(dfa_d, 3, accept_states={3})
    result_dfa = DFA.concat([dfa1, dfa2, dfa3], '~')
    
    assert result_dfa('aarb~r~lll') == (True, '<2>_3', True)
    return result_dfa

def test_sequence_dfa():
    seq = ["a", "b", "c", "a", "g"]
    dfa = DFA.from_sequence_ids(seq)
    assert dfa('abc') == (True, 3, False)
    assert dfa('abcag') == (True, 5, True)
    assert dfa('abagr')[0] == False
    assert dfa('abcagr')[0] == False

if __name__ == "__main__":
    test_DFA()
    test_concat_dfas()
    test_sequence_dfa()