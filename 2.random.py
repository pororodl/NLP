import numpy as np
nums = [[i] for i in range(10)]
nums = np.array(nums)
print(nums)
a = np.random.randint(0,10,3)
print(a)
print(nums[a])

