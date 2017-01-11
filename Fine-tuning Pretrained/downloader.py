import urllib
import io
import requests

f = open('people.txt')
i = 0
for line in f:
	temp = line.split('\n')[0]
	print temp
	try:
		if requests.head(temp, timeout=1).status_code == 200:
			urllib.urlretrieve(temp, 'in'+str(i)+'.jpg')
			i = i+1
	except:
		print 'Skip!\n'
