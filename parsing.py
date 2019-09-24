filename = "without_ls.txt"
a = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dic2 = {}
dic5 = {}
def parse(line):
    global dic2, dic5
    tmp = line.strip().split()
    #print tmp
    if tmp[0][0] in a:
        #if int(tmp[0]) <= 50 :
        #print (tmp[0], tmp[2], tmp[5])
        try :        
            #print (0)        
            dic2[int(tmp[0])].append(float(tmp[2]))
            dic5[int(tmp[0])].append(int(tmp[5]))
        except:
            dic2[int(tmp[0])]=[float(tmp[2])]
            dic5[int(tmp[0])]=[int(tmp[5])]
                
    return
    
with open(filename) as f:
    for line in f:
        parse(line)
    print dic2[0]
    for i in dic2:
        
        average = 0
        for a in dic2[i]:
            average += a
            #print (a)
        average /= 10.0     
        print (average)
    print (next)
    for i in dic5:
        average = 0
        for a in dic5[i]:
            average += a
            #print (a)
        average /= 10.0     
        print (average)