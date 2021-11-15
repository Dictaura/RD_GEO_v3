import RNA
from utils.rna_lib import get_distance_from_graph, edge_distance_norm, get_edge_h, get_graph, get_pair_ratio, \
    random_init_sequence_pair

seq = 'UACACAGAAUAGGGUCUUCCCACGCGACCGGGACAGGAACGAAACUAAAAGACCGACAGGUGACGCGGCAGUGGCCGCCCCGAAGCUCUUACCUCAGGACCCUGUUCUAAAGUAGCAACCUGUUUAGACAGGGUGGUCCUGGGGGUCAGCCCACAGAGGAGGUCCUCGGAGGCUGGCCCCAGCUCUAAGGGCCAUUUGAUGGCAGUACCAAUGUGGGUGUCCCCACAUUGUAGUUGCCAUUCACAGGGUGUC'
dotB = RNA.fold(seq)[0]

graph = get_graph(seq_base=seq, dotB=dotB)

ratio1 = get_pair_ratio(graph, 4)

seq, onehot = random_init_sequence_pair(dotB, graph.edge_index, max_size=len(dotB), action_space=4)

graph.x = onehot
graph.y['seq_base'] = seq

ratio2 = get_pair_ratio(graph, 4)

print(ratio2)
