
def QS_DSver_partition(arr,i,j,pk):
    origin_i = i
    origin_j = j
    i = i + 1 

    while(True):
        while(True):
            if(arr[i] >= arr[pk]):
                # print("i found:%d" % arr[i])
                # for data in arr: print(data, end=" ")
                # print("\n")
                break
            i+=1
        while(True):
            if(arr[j] <= arr[pk]):
                # print("j found:%d" % arr[j])
                # for data in arr: print(data, end=" ")
                # print("\n")
                break
            j-=1 
  
        if(i > j):
            SWAP(arr,pk,j) # pk SWAP with j, j be a new pk
            # print("pk to j!")
            # for data in arr: print(data, end=" ")
            # print("\n=============\n")
            return j
        else:
            SWAP(arr,i,j)
            # print("swap!")
            # for data in arr: print(data, end=" ")
            # print("\n")
            i+=1
            j-=1
        
def QuickSort_DS(arr,i,j,pk):
    if(i < j):
        pk = QS_DSver_partition(arr,i,j,pk) #get index 5
        QuickSort_DS(arr,i,pk-1,i)
        QuickSort_DS(arr,pk+1,j,pk+1)
  
    
def SWAP(arr,i,j):
    temp = arr[j]
    arr[j] = arr[i] 
    arr[i] = temp
    
if __name__ == "__main__":
    arr = [26,5,37,1,61,11,59,15,48,19] # 10 items
    QuickSort_DS(arr,0,len(arr)-1,0)
    for n in arr: print(n, end=" ") #result