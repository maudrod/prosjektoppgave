import scipy
import matplotlib.pyplot as plt
import numpy as np
import numba

xlist = np.linspace(0,10,50)
ylist = np.random.randint(0,100,50)

print(xlist)
print(ylist)

np.save("x_test",xlist)
np.save("y_test",ylist)

plt.plot(xlist,ylist)
plt.xlabel('Maud')
plt.ylabel('Emil')
plt.savefig('random_xy')
plt.show()



#with open('some_file.txt', 'w') as f:
#    f.write("x  %d" % x)
