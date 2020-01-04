import numpy as np
f = open('annot.txt','r+')
data = f.readlines()
g = open('ucf_labels.txt','w+')
for line in data:
    a = line.split()
    a[0] = str(int(a[0]) - 1)
    x = a[0] + ' ' + a[1] + '\n'
    g.writelines(x)
f.close()
g.close() 