import nltk
import numpy as np
#f = open("autuencoder.py","r")
ff = open("f:\\test1","r")
f0 = open("f:\\ref.0","r",encoding="utf-8")
f1 = open("f:\\ref.1","r",encoding="utf-8")
f2 = open("f:\\ref.2","r",encoding="utf-8")
f3 = open("f:\\ref.3","r",encoding="utf-8")
f4 = open("f:\\ref.4","r",encoding="utf-8")
f5 = open("f:\\ref.5","r",encoding="utf-8")
f6 = open("f:\\ref.6","r",encoding="utf-8")
f7 = open("f:\\ref.7","r",encoding="utf-8")
l = []

for line in ff:
    l.append(line.rstrip("\n"))
r0 = []
for line in f0:
    r0.append(line.rstrip("\n"))
r1 = []
for line in f1:
    r1.append(line.rstrip("\n"))
r2 = []
for line in f2:
    r2.append(line.rstrip("\n"))
r3 = []
for line in f3:
    r3.append(line.rstrip("\n"))
r4 = []
for line in f4:
    r4.append(line.rstrip("\n"))
r5 = []
for line in f5:
    r5.append(line.rstrip("\n"))
r6 = []
for line in f6:
    r6.append(line.rstrip("\n"))
r7 = []
for line in f7:
    r7.append(line.rstrip("\n"))

bleu = []
for i in range(len(l)):
    s = l[i].split(" ")
    s0 = r0[i].split(" ")
    s1 = r1[i].split(" ")
    s2 = r2[i].split(" ")
    s3 = r3[i].split(" ")
    s4 = r4[i].split(" ")
    s5 = r5[i].split(" ")
    s6 = r6[i].split(" ")
    s7 = r7[i].split(" ")
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([s0,s1,s2,s3,s4,s5,s6,s7], s)
    bleu.append(BLEUscore)
print(np.mean(bleu))