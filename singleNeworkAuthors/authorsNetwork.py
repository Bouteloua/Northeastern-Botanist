# #!/usr/bin/env python
# """
# An example using Graph as a weighted network.
# """
# __author__ = """Brian Franzone (Franzone89@gmail.com)"""
from pandas import *
import re, cPickle
from itertools import combinations
import networkx as nx
try:
    import matplotlib.pyplot as plt
except:
    raise
import matplotlib
matplotlib.style.use('ggplot')

import numpy as np
import pysal
import sys
import graphs
from matplotlib import cm


# for i in xrange(1800,2030,10):
#      print i - 10, i

def topTable(field1, field2, n_top):
    topM = max(field2) * 0.9
    right = len(field1) * 0.75
    plt.text(right, topM * 1.08, 'Top %s' % n_top, fontsize=12)
    for i in range(n_top):
        curr = field1[i]
        val = field2[i]
        plt.text(right, topM - i * topM / 20, '{}) {} = {}'.format(i + 1,
        curr.upper(), round(val, 3)), fontsize=8)

df = read_csv('acc_names_plantList.csv')
collectorNameCol = 'edited'


df = df.dropna(subset=[collectorNameCol], how='all')


G=nx.Graph()

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(1,1,1)
ax.patch.set_facecolor('black')
ax.set_axis_bgcolor('black')
collectorNodeDic = {}
collectorEdgeDic = {}
collectorList = []

for DATA in df[collectorNameCol]:
	collectorList.append(filter(None, re.split("[,;&%()]+", DATA)))

#Weight for nodes
for i in collectorList:
	for j in i:

		# print j.strip()
		if len(j.strip()) > 1 and '.' in j:
			collectorNodeDic[j.strip().lower().replace(' ' , '')] = 0

for i in collectorList:
	for j in i:
		if len(j.strip()) > 1 and '.' in j:
			collectorNodeDic[j.strip().lower().replace(' ' , '')] += 1


#Weight for edges
for L1 in collectorList:
	for combo in combinations(L1, 2):
		if len(combo[0].strip()) > 1 and len(combo[1].strip()) > 1 and '.' in combo[0] and '.' in combo[1]:
			collectorEdgeDic[combo[0].strip().lower().replace(' ' , '') +','+combo[1].strip().lower().replace(' ' , '')] = 0

for L1 in collectorList:
	for combo in combinations(L1, 2):
		if len(combo[0].strip()) > 1 and len(combo[1].strip()) > 1 and '.' in combo[0] and '.' in combo[1]:
			collectorEdgeDic[combo[0].strip().lower().replace(' ' , '') +','+combo[1].strip().lower().replace(' ' , '')] += 1


for L1 in collectorList:
	for combo in combinations(L1, 2):
		if len(combo[0].strip()) > 1 and len(combo[1].strip()) > 1 and '.' in combo[0] and '.' in combo[1]:
			# print combo[0].strip().lower() +','+ combo[1].strip().lower()
			G.add_edge(combo[0].strip().lower().replace(' ' , ''), combo[1].strip().lower().replace(' ' , ''), weight=collectorEdgeDic[combo[0].strip().lower().replace(' ' , '') +','+ combo[1].strip().lower().replace(' ' , '')])

nodeList = []
for i in nx.nodes_iter(G):
	nodeList.append(i)

with open('nodelist/fullNodeList.pickle', 'wb') as f:
	cPickle.dump(collectorNodeDic, f)

# print collectorNodeDic


remove = [node for node,degree in G.degree().items() if degree < -1]

# remove1 = [degree for node,degree in G.degree().items()]

# print remove1
# sys.exit()

collSpecimenCount_temp = []
for k, v in collectorNodeDic.iteritems():
	collSpecimenCount_temp.append(v)
collSpecimenCount = np.array(collSpecimenCount_temp)

collSpecimenBreaks = pysal.Natural_Breaks(collSpecimenCount, k=4)

print collSpecimenBreaks

# greater1000Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > collSpecimenBreaks.bins[2] and k in nodeList and k not in remove]
# greater500Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > collSpecimenBreaks.bins[1] and r < collSpecimenBreaks.bins[2] and k in nodeList and k not in remove]
# greater100Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > collSpecimenBreaks.bins[0] and r < collSpecimenBreaks.bins[1] and k in nodeList and k not in remove]
# less100Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r < collSpecimenBreaks.bins[0] and k in nodeList and k not in remove]
# # if r['x'] > 92 and



# elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >5]

# print elarge


collEdgeCount_temp = [d['weight'] for (u,v,d) in G.edges(data=True) if u not in remove and v not in remove]


collEdgeCount = pysal.Natural_Breaks(np.array(collEdgeCount_temp), k=2)

slarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 200 and u not in remove and v not in remove]

elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > collEdgeCount.bins[0] and u not in remove and v not in remove]


esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= collEdgeCount.bins[1] and u not in remove and v not in remove]

# print esmall
# print remove
# # sys.exit()
# pos=nx.spring_layout(G, k=.5,iterations=20, weight=20) # positions for all nodes
pos = nx.graphviz_layout(G, prog='neato')


# pos=nx.spring_layout(G, weight=20) # positions for all nodes
# pos=nx.circular_layout(G) # positions for all nodes

G.remove_nodes_from(remove)
# cm.Spectral(betc_value)
node_color = [cm.bwr(G.degree(v)) for v in G]
# nodes
nx.draw_networkx_nodes(G,pos, node_size=[float(G.degree(v)) * 2.5 for v in G], alpha=0.6, node_color=node_color)
# nx.draw_networkx_nodes(G,pos,nodelist=greater500Coll, node_size=[float(G.degree(v)) * 2.5 for v in G],alpha=0.6,node_color="#fdae61", with_labels=True, label=str(collSpecimenBreaks.bins[1]) + ' - ' + str(collSpecimenBreaks.bins[2] - 1))
# nx.draw_networkx_nodes(G,pos,nodelist=greater100Coll, node_size=[float(G.degree(v)) * 2.5 for v in G],alpha=0.6,node_color="#5e3c99", with_labels=True, label=str(collSpecimenBreaks.bins[1] -1) + ' - ' + str(collSpecimenBreaks.bins[0]))
# nx.draw_networkx_nodes(G,pos,nodelist=less100Coll,node_size=[float(G.degree(v)) * 2.5 for v in G],alpha=0.6,node_color="#b2abd2", with_labels=True, label='<= ' + str(collSpecimenBreaks.bins[0] - 1))
print [float(G.degree(v)) * .8 for v in G]

nx.draw_networkx_edges(G,pos,edgelist=slarge,
                    width=[float(G.degree(v)) * .8 for v in G],alpha=0.6,edge_color='#E71D36', style='solid', label=str(collEdgeCount.bins[1]) + ' - ' + str(collEdgeCount.bins[0] + 1))

nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=.8,alpha=.4,edge_color='#FF9F1C', style='solid', label=str(collEdgeCount.bins[1]) + ' - ' + str(collEdgeCount.bins[0] + 1))
# edges
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=.7,alpha=0.3,edge_color='#32936F', style='dashed' , label='<= ' + str(collEdgeCount.bins[0]))

print collEdgeCount
# nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                     width=.5,alpha=0.4,edge_color='b',style='dashed')

# labels
# nx.draw_networkx_labels(G,pos,font_size=2,font_family='sans-serif', alpha=0.5)
# print G.nodes('4')
# L= plt.legend(G.nodes())
# plt.legend(frameon=False)
# # plt.title('Connectivity in Natural History Collections' , fontsize=20, loc='center')
# plt.legend(title="# of collections per individual and shared connectivity with others", ncol=3, shadow=True, fancybox=True, fontsize=14, loc='lower center', columnspacing=1).get_frame().set_facecolor('#FFFFFF')

# for i in  L.get_texts():
# 	print i
# t = np.arange(0., 5., 0.2)
# lines = ax.plot(t, t, 'r-', t, t**2, 'bs', t, t**3, 'g^')
# figlegend.legend(lines, ('one', 'two', 'three', 'four'), 'lower left')
# figlegend.savefig('legend.png')
#key
#################
filename11 = 'edgelist/master_list.edgelist'
nx.write_edgelist(G,filename11, delimiter=',', data=['weight'])
######################
plt.axis('off')
plt.savefig("New_England_Collector_Network.png", dpi=100) # save as png
plt.show() # display

stats = graphs.graphStats()
betc_key, betc_value = stats.calculate_betweenness_centrality(G)
degc_key, degc_value = stats.calculate_degree_centrality(G)

#Average degree
N = G.order()
K = G.size()
avg_d = float(N) / K
avg_degree = 'Average degree: %.4f ' % (avg_d)


# Plot: Degree_centrality
plt.figure()

ax1 = plt.subplot(211)
plt.title('Degree centrality for nodes', fontsize=12)
a_lenght = np.arange(len(degc_value))
plt.bar(a_lenght[:50], degc_value[:50], color=cm.Spectral(degc_value), align='center')
plt.xticks(a_lenght, degc_key, size='small', rotation=45)
plt.tick_params(axis='x', labelsize=4)
plt.tick_params(axis='y', labelsize=8)
plt.autoscale(enable=True, axis='both', tight=None)

#Top degree centrality:
topTable(degc_key, degc_value, 10)
plt.text(len(degc_value) * 0.75, max(degc_value) * 0.4, avg_degree,
bbox={'facecolor': 'blue', 'alpha': 0.25, 'pad': 10}, fontsize=7)

# Plot: Betweenness_centrality
plt.subplot(212)
plt.title('Betweenness centrality for nodes', fontsize=12)
a_lenght = np.arange(len(betc_value))
plt.bar(a_lenght[:50], betc_value[:50], color=cm.Spectral(betc_value), align='center')
plt.xticks(a_lenght, betc_key, size='small', rotation=45)
plt.tick_params(axis='x', labelsize=5)
plt.tick_params(axis='y', labelsize=8)
plt.autoscale(enable=True, axis='both', tight=None)
plt.ylim(0, max(betc_value) * 1.1)
plt.plot(betc_value[:50], '--b')

#Top degree betweenness:
topTable(betc_key, betc_value, 10)
plt.savefig("NE_centrality.png", dpi=400) # save as png

plt.show()



# gbifID -> 0
# catalogNumber -> 65
# collectionCode -> 67
# continent -> 69
# countryCode -> 70
# stateProvince -> 177
# recordedBy -> 168
# recordNumber -> 167
# year -> 194
# family -> 92
# genus -> 99
# specificEpithet -> 175
# species -> 218




# id -> 1
# dwc:catalogNumber -> 5
# dwc:recordedBy -> 10
# dwc:recordNumber -> 64
# dwc:country -> 14
# dwc:kingdom -> 45
# dwc:family-> 26
# dwc:genus -> 34
# dwc:specificEpithet ->67
# dwc:stateProvince -> 69
# dwc:eventDate -> 25

# cut -d',' -f1,5,10,14,,25,26,30,34,45,65,67,69 head.csv > outfile.csv
