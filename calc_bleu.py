import nltk
f1 = open("/data/wtm/data/wikilarge/wiki.full.aner.ori.valid.dst",'r')
f2 = open("data/wtm/data/wikilarge/1",'r')
for i in range(len(f1)):
    line1 = f1[i]
    line2 = f2[i]
    print(line1,line2)