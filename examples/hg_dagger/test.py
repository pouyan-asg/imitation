import numpy as np

arr = [False, False , False]

print(np.any(arr))
print(np.sum(arr) != 0)

is_takeover = False

# For environments where human is NOT intervening, use the learner's action
if np.any(~is_takeover):
    print("yes")