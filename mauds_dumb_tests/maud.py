
import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3, 4], [10, 3, 5, 9, 11])

plt.xlabel('Months')
plt.ylabel('Books Read')
plt.show()

x = 5

print(x)

with open('some_file.txt', 'w') as f:
    f.write("x  %d" % x)
