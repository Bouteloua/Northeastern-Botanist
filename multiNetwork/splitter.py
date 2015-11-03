from pandas import *

df = read_csv('full_idigbio_Oct_12_2015_year.csv')

for i in xrange(1800,2016):
	temp = df[(df['year'] == i)]
	filename = 'inputFiles/year_' + str(i) + '.csv'
	# temp.to_csv(filename, index = False, header=False)
	temp.to_csv(filename, index = False, header=True)