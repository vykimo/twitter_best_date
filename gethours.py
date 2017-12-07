import json
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import numpy as np
import calendar

def get_hoursdays(fname):
	data = json.load(open(fname))
	
	x1 = []
	x2 = []
	x3 = []
	days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
	y = []
	for tweet in data:
		y.append(tweet['rt'] * 2 + tweet['fav'])
		hour = datetime.strptime(tweet['date'], '%a %b %d %H:%M:%S %z %Y')
		x1.append(int(hour.strftime('%H%M')))
		x2.append(int(hour.strftime('%w')))
		x3.append(int(hour.strftime('%m%d')))
		
			
	# Part 1 : Hour repartition
	plt.figure(1)
	plt.hist(x1, 100, normed=1, facecolor='green', alpha=0.5)
	plt.title("Hours repartition")
	plt.xlabel('Hours')
	plt.ylabel('Repartition')
	plt.legend()

	# Part 2 : hours pop		
	plt.figure(2)
	plt.plot(x1,y,'.',color='green')
	plt.title("Hours popularity")
	plt.xlabel('Hours')
	plt.ylabel('Popularity')
	plt.legend()
	
	
	#label_unique = list(set(x2))
	#x5 = [0] * len(label_unique)
	#for i in range(0,len(label_unique)):
	#	for j in range(0,len(x2)):
	#		if x2[j] == label_unique[i]:
	#			x5[i]+=1
			
	#plt.figure(5)
	#plt.plot(label_unique,x5,'x')
	#plt.title("Weekday repartition")
	#plt.xlabel('Weekday')
	#plt.ylabel('Repartition (%)')
	#plt.legend()
	
	
	# Part 3 : Weekday repartition
	plt.figure(3)
	plt.hist(x2, 100, normed=1, facecolor='orange', alpha=0.5)
	plt.title("Weekday repartition")
	plt.xlabel('Weekday')
	plt.ylabel('Repartition')
	plt.legend()

	# Part 4 : Weekday pop		
	plt.figure(4)
	
	label_unique = list(set(x2))
	
	total = []
	for i in range(0,len(label_unique)):	
		x_total = 0
		x_count = 0
		for j in range(0,len(x2)):
			if label_unique[i] == x2[j]:
				x_total += y[j]
				x_count += 1
		total.append(x_total/x_count)
	
	plt.plot(total,'x-')
	plt.plot(x2,y,'.',color='orange')
	plt.title("Weekday popularity")
	plt.xlabel('Weekday')
	plt.ylabel('Popularity')
	plt.legend()
	
	# Part 5 : Day repartition
	plt.figure(5)
	plt.hist(x3, 100, normed=1, facecolor='b', alpha=0.5)
	plt.title("Day repartition")
	plt.xlabel('Day')
	plt.ylabel('Repartition')
	plt.legend()

	# Part 6 : Day pop	

	for m in range(1,13):
		x_temp = []
		y_temp = []
		for i in range(0,len(x3)):
			if int(x3[i]) >= m*100 and int(x3[i]) <= (m*100)+100:
				x_temp.append(x3[i])
				y_temp.append(y[i])
			
		plt.figure(5)
		plt.subplot(3,4,m)
		plt.hist(x_temp, 100, normed=1, facecolor='b', alpha=0.5)	
		plt.title(calendar.month_name[m])
		plt.figure(6)
		plt.subplot(3,4,m)
		plt.plot(x_temp,y_temp,'.')		
		plt.title(calendar.month_name[m])
	plt.legend()
	
	plt.title("Day popularity")
	
	plt.show()
	