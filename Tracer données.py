"""Tracer les données HYP"""


import matplotlib.pyplot as plt
import csv

file = open("test.csv")
csvreader = csv.reader(file)
hyp = []
t = []
#for n in range(100):
for row in csvreader:
    t.append(row[0])
    hyp.append(row[1])
file.close()

plt.figure(1)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.scatter(t[1:10], hyp[1:10])
#ax = plt.gca()
#ax.get_xaxis().set_visible(False)
plt.xticks(rotation=45)
plt.show()
print('ok')
