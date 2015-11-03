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
import glob
import graphs

def topTable(field1, field2, n_top):
    topM = max(field2) * 0.9
    right = len(field1) * 0.75
    plt.text(right, topM * 1.08, 'Top %s' % n_top, fontsize=4)
    for i in range(n_top):
        curr = field1[i]
        val = field2[i]
        plt.text(right, topM - i * topM / 20, '{}) {} = {}'.format(i + 1,
        curr.upper(), round(val, 3)), fontsize=4)

def nodeColors(G):
	node_color = []
	for v in G:
		if G.degree(v) > 15:
			node_color.append('#AB0000')
		elif G.degree(v) <= 15 and G.degree(v) >= 5:
			node_color.append('#D75813')
		else:
			node_color.append('#007EA7')
	return node_color

def edgeLines(G):
	node_color = []
	for v in G:

		if G.degree(v) > 10:
			print v, G.degree(v), 'SOLID!!!!'
			node_color.append('solid')
		else:
			print v, G.degree(v)
			node_color.append('dotted')
	return node_color

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

filenames = glob.glob('inputCSVs/*.csv')
print filenames
for filename in filenames:
	filename = filename.split('/')[1:][0]
	# for i in xrange(1800,2030,10):
	#      print i - 10, i
	collectorNameCol = 'edit_recordedBy'

	df = read_csv('inputCSVs/' + filename)



	df = df.dropna(subset=[collectorNameCol], how='all')


	G=nx.Graph()

	fig = plt.figure(figsize=(10,5))
	fig.patch.set_facecolor('black')
	ax = fig.add_subplot(1,1,1)

	collectorNodeDic = {}
	collectorEdgeDic = {}
	collectorList = []

	for DATA in df[collectorNameCol]:
		collectorList.append(filter(None, re.split("[,;&]+", DATA)))

	#Weight for nodes
	for i in collectorList:
		for j in i:
			if len(j.strip()) > 4 and '.' in j:
				collectorNodeDic[j.strip().lower()] = 0

	for i in collectorList:
		for j in i:
			if len(j.strip()) > 4 and '.' in j:
				collectorNodeDic[j.strip().lower()] += 1


	#Weight for edges
	for L1 in collectorList:
		for combo in combinations(L1, 2):
			if len(combo[0].strip()) > 4 and len(combo[1].strip()) > 4 and '.' in combo[0] and '.' in combo[1]:
				collectorEdgeDic[combo[0].strip().lower() +','+combo[1].strip().lower()] = 0

	for L1 in collectorList:
		for combo in combinations(L1, 2):
			if len(combo[0].strip()) > 4 and len(combo[1].strip()) > 4 and '.' in combo[0] and '.' in combo[1]:
				collectorEdgeDic[combo[0].strip().lower() +','+combo[1].strip().lower()] += 1


	for L1 in collectorList:
		for combo in combinations(L1, 2):
			if len(combo[0].strip()) > 4 and len(combo[1].strip()) > 4 and '.' in combo[0] and '.' in combo[1]:
				G.add_edge(combo[0].strip().lower(),combo[1].strip().lower(),weight=collectorEdgeDic[combo[0].strip().lower() +','+combo[1].strip().lower()])

	nodeList = []
	for i in nx.nodes_iter(G):
		nodeList.append(i)

	with open('nodelist/'+ filename[5:-4] +'_NodeList.pickle', 'wb') as f:
		cPickle.dump(collectorNodeDic, f)


	remove = [node for node,degree in G.degree().items() if degree < 1]

	# remove1 = [degree for node,degree in G.degree().items()]

	# print remove1
	# sys.exit()

	collSpecimenCount_temp = []
	for k, v in collectorNodeDic.iteritems():
		collSpecimenCount_temp.append(v)
	collSpecimenCount = np.array(collSpecimenCount_temp)
	try:
		collSpecimenBreaks = pysal.Natural_Breaks(collSpecimenCount, k=4)
		print collSpecimenBreaks
	except:
		continue


	# greater1000Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > 200 and k in nodeList and k not in remove]
	# greater500Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > 100 and r < 200 and k in nodeList and k not in remove]
	# greater100Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r > 50 and r < 100 and k in nodeList and k not in remove]
	# less100Coll = [ (k) for k,r in collectorNodeDic.iteritems() if r < 10 and k in nodeList and k not in remove]
	# # if r['x'] > 92 and



	# elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >5]

	# print elarge


	collEdgeCount_temp = [d['weight'] for (u,v,d) in G.edges(data=True) if u not in remove and v not in remove]

	print np.sum(collEdgeCount_temp)
	try:
		collEdgeCount = pysal.Natural_Breaks(np.array(collEdgeCount_temp), k=2)
		print collEdgeCount
	except:
		continue

	# collEdgeCount = pysal.Natural_Breaks(np.array(collEdgeCount_temp), k=4)
	# print collEdgeCount
	threeHuge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 100 and u not in remove and v not in remove]

	twoLarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 50 and d['weight'] <= 100 and u not in remove and v not in remove]


	oneSmallest=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= 49 and u not in remove and v not in remove]

	# print esmall
	# print remove
	# # sys.exit()
	pos=nx.spring_layout(G, k=.2,iterations=20, weight=20) # positions for all nodes
	# pos=nx.spring_layout(G, weight=20) # positions for all nodes
	# pos=nx.circular_layout(G) # positions for all nodes

	G.remove_nodes_from(remove)
	node_color = nodeColors(G)
	nodes = nx.draw_networkx_nodes(G,pos, node_size=[float(G.degree(v)) * 4.5 for v in G], alpha=0.6, node_color=node_color, with_labels=True, label="Amount of collected material")
	# nx.draw_networkx_nodes(G,pos,nodelist=greater1000Coll, node_size=10,alpha=0.6,node_color="#FFFF00", with_labels=True, label="hello")
	try:
		nodes.set_edgecolor(node_color)
	except:
		pass
	# nodes
	# nx.draw_networkx_nodes(G,pos,nodelist=greater1000Coll, node_size=200,alpha=0.6,node_color="#d7191c", with_labels=True, label= ' >= ' + str(200))
	# nx.draw_networkx_nodes(G,pos,nodelist=greater500Coll, node_size=150,alpha=0.6,node_color="#fdae61", with_labels=True, label=str(100) + ' - ' + str(199))
	# nx.draw_networkx_nodes(G,pos,nodelist=greater100Coll, node_size=75,alpha=0.6,node_color="#5e3c99", with_labels=True, label=str(50) + ' - ' + str(99))
	# nx.draw_networkx_nodes(G,pos,nodelist=less100Coll,node_size=35,alpha=0.6,node_color="#b2abd2", with_labels=True, label='<= ' + str(49))

	# edges
	nx.draw_networkx_edges(G,pos,edgelist=threeHuge,
                    width=[float(G.degree(v)) * 1 for v in G],alpha=0.6,edge_color='#E71D36', style='solid', label=str(200))

	nx.draw_networkx_edges(G,pos,edgelist=twoLarge,
                    width=.8,alpha=.4,edge_color='#FF9F1C', style='solid', label=str(100) + ' - ' + str(199))
	# edges
	nx.draw_networkx_edges(G,pos,edgelist=oneSmallest,
                    width=.7,alpha=0.2,edge_color='#32936F', style='dashed' , label='<= ' + str(99))


	# nx.draw_networkx_edges(G,pos,edgelist=esmall,
	#                     width=.5,alpha=0.4,edge_color='b',style='dashed')

	# labels
	nx.draw_networkx_labels(G,pos, font_color='#FFFFFF', font_size=4,font_family='sans-serif', alpha=0.5)
	# print G.nodes('4')
	# L= plt.legend(G.nodes())
	# plt.legend(frameon=False)
	# plt.title(filename[5:-4] , fontsize=30, loc='center')
	# plt.legend(title="# of collections per indiv. and shared connectivity", ncol=3, shadow=True, fancybox=True, fontsize=4, loc='lower center', columnspacing=15, handlelength=3, handleheight=4).get_frame().set_facecolor('#FFFFFF')

	# for i in  L.get_texts():
	# 	print i
	# t = np.arange(0., 5., 0.2)
	# lines = ax.plot(t, t, 'r-', t, t**2, 'bs', t, t**3, 'g^')
	# figlegend.legend(lines, ('one', 'two', 'three', 'four'), 'lower left')
	# figlegend.savefig('legend.png')
	#key
	#################
	filename11 = 'edgelist/' + filename[5:-4] + '.edgelist'
	nx.write_edgelist(G,filename11,delimiter=',',data=['weight'])
	######################
	plt.axis('off')
	plt.savefig('outputGraphs/graph_'+filename[5:-4] + '.jpg', dpi=300, facecolor=fig.get_facecolor()) # save as png
	# plt.show() # display
	if np.sum(collEdgeCount_temp) > 20:
		stats = graphs.graphStats()
		betc_key, betc_value = stats.calculate_betweenness_centrality(G)
		degc_key, degc_value = stats.calculate_degree_centrality(G)

		#Average degree
		N = G.order()
		K = G.size()
		avg_d = float(N) / K
		avg_degree = 'Average degree: %.4f ' % (avg_d)


		# Plot: Degree_centrality
		fig = plt.figure(figsize=(10,5))
		# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
		ax1 = plt.subplot(211)
		simpleaxis(ax1)
		# ax1.text(.5,.9,'Degree centrality for nodes for year ' + filename[5:-4],
        # horizontalalignment='center',
        # transform=ax1.transAxes)
		# plt.title('Degree centrality for nodes for year ' + filename[5:-4], fontsize=7)
		a_lenght = np.arange(len(degc_value))
		plt.bar(a_lenght[:50], degc_value[:50], color='#007EA7', align='center')
		plt.xticks(a_lenght, degc_key, size='small', rotation=35)
		plt.tick_params(axis='x', labelsize=4)
		plt.tick_params(axis='y', labelsize=4)
		plt.autoscale(enable=True, axis='both', tight=None)
		plt.plot(degc_value[:50], '--b', color='#AB0000')

		#Top degree centrality:
		try:
			topTable(degc_key[:50], degc_value[:50], 10)
		except:
			pass
		plt.text(len(degc_value[:50]) * 0.75, max(degc_value[:50]) * 0.4, avg_degree,
		bbox={'facecolor': '#D75813', 'alpha': 0.25, 'pad': 5}, fontsize=5)

		# Plot: Betweenness_centrality
		ax2 = plt.subplot(212)
		simpleaxis(ax2)
		# ax2.text(.5,.9,'Betweenness centrality for nodes for year ' + filename[5:-4],
        # horizontalalignment='center',
        # transform=ax2.transAxes)
		# plt.title('Betweenness centrality for nodes for year ' + filename[5:-4], fontsize=7)
		a_lenght = np.arange(len(betc_value))
		plt.bar(a_lenght[:50], betc_value[:50], color='#007EA7', align='center')
		plt.xticks(a_lenght, betc_key, size='small', rotation=35)
		plt.tick_params(axis='x', labelsize=4)
		plt.tick_params(axis='y', labelsize=4)
		plt.autoscale(enable=True, axis='both', tight=None)
		plt.ylim(0, max(betc_value) * 1.1)
		plt.plot(betc_value[:50], '--b', color='#AB0000')
		#Top degree betweenness:
		try:
			topTable(betc_key[:50], betc_value[:50], 10)
		except:
			pass
		plt.tight_layout()
		titleName = 'outputStats/stats_'+filename[5:-4] + '.jpg'
		plt.savefig(titleName, dpi=200) # save as png


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
