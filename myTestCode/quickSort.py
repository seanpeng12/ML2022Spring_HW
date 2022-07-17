
def QuickSort_DSver(arr,i,j,pk):
    origin_i = i
    origin_j = j
    i = i + 1 

    while(True):
        while(True):
            if(arr[i] >= arr[pk]):
                print("i found:%d" % arr[i])
                break
            i+=1
        while(True):
            if(arr[j] <= arr[pk]):
                print("j found:%d" % arr[j])
                break
            j-=1 
        print("i= %d;j= %d" % (i, j))   
        if(j < i):
            arr [j] = arr[0]  # pk data place to j
            # QuickSort_DSver(arr,origin_i,j-1,arr[])
            # QuickSort_DSver(arr,i,origin_j,arr[])
            break
        else:
            i+=1
            j-=1
        
        
        

arr = [26,5,37,1,61,11,59,15,48,19] # 10 items
QuickSort_DSver(arr,0,9,0)