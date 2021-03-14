import numpy as np
print(np.char.add(['hello', 'hi'],[' abc', ' xyz']))# ['hello abc' 'hi xyz']
print (np.char.multiply(['Run ','ni hao '],[3,2]))# ['Run Run Run ' 'ni hao ni hao ']
print(np.char.center(['ni','hao','ma'],5,['a','b','c']))# ['aania' 'bhaob' 'ccmac']
print(np.char.capitalize(['my name is qi','hello hello']))# ['My name is qi' 'Hello hello']
print(np.char.title(['my name is qi','hello hello']))# ['My Name Is Qi' 'Hello Hello']
print(np.char.upper(['my name is qi','hello hello']))#['MY NAME IS QI' 'HELLO HELLO']
print(np.char.lower(['MY NAME IS QI','HELLO HELLO']))#['my name is qi' 'hello hello']
print(np.char.split(['ni hao','how it going'],' '))# [list(['ni', 'hao']) list(['how', 'it', 'going'])]

