
def snake_seq(M, N):
    seq_list=[]
    for i in range(M):
        if i%2==0:
            rj=range(N)
        else:
            rj=range(N-1,-1,-1)
        for j in rj:
            seq_list.append([i,j])
    return seq_list
    