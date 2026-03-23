import networkx as nx

file = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/DDI0413D_cortexm1_r1p0_trm/graph_rag/output/20240723-095058/artifacts/clustered_graph.0.graphml'

g = nx.read_graphml(file)
print(g)
print(list(g.nodes(data=True))[0])