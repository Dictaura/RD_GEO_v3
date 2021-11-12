import RNA
from utils.rna_lib import get_distance_from_graph, edge_distance_norm, get_edge_h

dotB1 = '(((((((....(((...........)))((((((((..(((((((((((((((((((...(((((......))))).)))))).)))))))))))))..))))))))..)))))))'
dotB2 = '((((((.((((....))))))).)))..........'

edge1_h = get_edge_h(dotB1)
edge2_h = get_edge_h(dotB2)

l1 = len(dotB1)
l2 = len(dotB2)

distance = edge_distance_norm(edge1_h, edge2_h, l1, l2)
