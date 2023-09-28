import numpy as np

# create an array NumPy
my_array = np.array([1, 2, 3, 4, 5])

# print the array
print("array:")
print(my_array)

# calculate the average value of an array
mean_value = np.mean(my_array)
print("average value:", mean_value)

# finding max and min elements of the array 
max_value = np.max(my_array)
min_value = np.min(my_array)
print("max_value:", max_value)
print("min_value:", min_value)

# calculate the squares of each element of the array
squared_array = np.square(my_array)
print("square of the array:")
print(squared_array) 