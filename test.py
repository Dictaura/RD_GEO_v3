import RNA

seq1 = 'UACACAGAAUAGGGUCUUCCCACGCGACCGGGACAGGAACGAAACUAAAAGACCGACAGGUGACGCGGCAGUGGCCGCCCCGAAGCUCUUACCUCAGGACCCUG'
seq2 = 'UUCUAAAGUAGCAACCUGUUUAGACAGGGUGGUCCUGGGGGUCAGCCCACAGAGGAGGUCCUCGGAGGCUGGCCCCAGCUCUAAGGGCCAUUUGAUGGCAGUACCAAUGUGGGUGUCCCCACAUUGUAGUUGCCAUUCACAGGGUGUC'
dotB = RNA.cofold(seq1, seq2)
dotB2 = RNA.fold(seq1)

# graph = get_graph(seq_base=seq, dotB=dotB)
#
# ratio1 = get_pair_ratio(graph, 4)
#
# seq, onehot = random_init_sequence_pair(dotB, graph.edge_index, max_size=len(dotB), action_space=4)
#
# graph.x = onehot
# graph.y['seq_base'] = seq
#
# ratio2 = get_pair_ratio(graph, 4)

print(1)
