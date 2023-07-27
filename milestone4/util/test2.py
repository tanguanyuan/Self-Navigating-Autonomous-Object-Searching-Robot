# # Python Program illustrating
# # numpy.mean() method
# import numpy as np
	

# # 2D array
# # arr = [[[11, 17, 12, 33, 44],
# # 	[15, 6, 27, 8, 19],
# # 	[23, 2, 54, 1, 4, ]],
# #     [[14, 17, 12, 33, 44],
# # 	[15, 6, 27, 8, 19],
# # 	[23, 2, 54, 1, 4, ]],
# #     [[14, 17, 12, 33, 44],
# # 	[15, 6, 27, 8, 19],
# # 	[23, 2, 54, 1, 4, ]]]
# # print("arr2 ", arr[0:])
# # print("\nmean of arr, axis = 0 : ", np.mean(arr, axis = 0))

# # mean of the flattened array
# # print("\nmean of arr, axis = None : ", np.mean(arr))
	
# # # mean along the axis = 0
# # print("\nmean of arr, axis = 0 : ", np.mean(arr, axis = 0))

# # # mean along the axis = 1
# # print("\nmean of arr, axis = 1 : ", np.mean(arr, axis = 1))

# # out_arr = np.arange(3)
# # print("\nout_arr : ", out_arr)
# # print("mean of arr, axis = 1 : ",
# # 	np.mean(arr, axis = 1, out = out_arr))

# # 2D array 
# arr = [[14, 17, 12, 33, 44],  
#        [15, 6, 27, 8, 19], 
#        [23, 2, 54, 1, 4, ]] 
    
# # mean of the flattened array 
# print("\nmean of arr, axis = None : ", np.mean(arr)) 
    
# # mean along the axis = 0 
# print("\nmean of arr, axis = 0 : ", np.mean(arr[0:][0:], axis = 0)) 
   
# # mean along the axis = 1 
# print("\nmean of arr, axis = 1 : ", np.mean(arr, axis = 1))
  
# out_arr = np.arange(3)
# print("\nout_arr : ", out_arr) 
# print("mean of arr, axis = 1 : ", 
#       np.mean(arr, axis = 1, out = out_arr))

from scipy import stats
x2 = [[ 2,  1,  2,  3],
       [ 19,  5,  6,  7],
       [ 20,  9, 10, 11],
       [22, 13, 14, 15],
       [78, 17, 18, 19]]

print(stats.trim_mean(x2, 0.25))
#print(stats.trim_mean(x2, 0.25, axis=1))
