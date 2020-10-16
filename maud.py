
import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])

plt.xlabel('Months')
plt.ylabel('Books Read')
plt.savefig('books_read.png')

x = 5

print(x)

with open('some_file.txt', 'w') as f:
    f.write("x  %d" % x)
