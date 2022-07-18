
def QS_DSver_partition(arr,i,j,pk):
    origin_i = i
    origin_j = j
    i = i + 1 

    while(True):
        while(True):
            if(arr[i] >= arr[pk]):
                print("i found:%d" % arr[i])
                for data in arr: print(data, end=" ")
                print("\n")
                break
            i+=1
        while(True):
            if(arr[j] <= arr[pk]):
                print("j found:%d" % arr[j])
                for data in arr: print(data, end=" ")
                print("\n")
                break
            j-=1 
  
        if(i > j):
            SWAP(arr,pk,j) # pk SWAP with j, j be a new pk
            for data in arr: print(data, end=" ")
            print("\n")
            # return origin_i,origin_j, j
            return j
        else:
            SWAP(arr,i,j)
            for data in arr: print(data, end=" ")
            print("\n")

            i+=1
            j-=1
        
def QuickSort_DS(arr,i,j,pk):
    # print()
    if(i < j):
        pk = QS_DSver_partition(arr,i,j,pk) #get index 5
        print(pk)
        # origin_i = res[0]
        # origin_j = res[1]
        # new_pk = res[2]

        QuickSort_DS(arr,i,pk-1,arr[i])
        QuickSort_DS(arr,pk+1,j,arr[pk+1])


        # res1 = QS_DSver_partition(arr,0,4,0)    6 9 6
        # print(res1)
        # res2 = QS_DSver_partition(arr,0,1,0) 
        # print(res2)

    

  
    
def SWAP(arr,i,j):
    temp = arr[j]
    arr[j] = arr[i] 
    arr[i] = temp
    
if __name__ == "__main__":
    arr = [26,5,37,1,61,11,59,15,48,19] # 10 items
    res = QuickSort_DS(arr,0,9,0)