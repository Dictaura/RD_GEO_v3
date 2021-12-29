import RNA

from utils.rna_lib import get_graph, random_init_sequence_pair, get_pair_ratio

dataset = ['...((((((((...)))))((.........))................((((...(((((((((((....))))))..)))))...)))).(((..((.(((....(((((..((....)).)))))....)))...))..)))..(((((..(.((((....)))))..)))))(((((.......)))))....((((((((.(((.((((((((....)))))))))))..))))).))).....))).']

graph = get_graph(dotB=dataset[0])

graph = random_init_sequence_pair(graph.y['dotB'], graph.edge_index, max_size=len(dataset[0]), action_space=4)

ratio = get_pair_ratio(graph, action_space=4)

print(ratio)