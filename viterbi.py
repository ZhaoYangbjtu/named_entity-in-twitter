import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []
    protable=np.zeros((L,N))
    backtable=np.zeros((L,N))
    for i in xrange(N):
        if i==0:
            for j in xrange(L):
                protable[j][i]=start_scores[j]+emission_scores[i][j]

        else:
            for j in xrange(L):
                maxscore=-float('inf')
                for k in xrange(L):
                    temp=protable[k][i-1]+trans_scores[k][j]+emission_scores[i][j]
                    if temp>maxscore:
                        maxscore=temp
                        protable[j][i]=maxscore
                        backtable[j][i]=k
    #
    maxdict={}
    for i in xrange(L):
        maxscore1=-float('inf')
        mark=0
        temp=protable[i][N-1]+end_scores[i]
        if temp>maxscore1:
            maxscore1=temp
            mark=i
        maxdict[maxscore1]=i
    value=max(maxdict.keys())
    # backtrace
    sqlist = []
    n = N-1
    index = maxdict[value]
    sqlist.append(index)
    while n != 0:
        temp = backtable[index][n]
        sqlist.append(int(temp))
        index = int(temp)
        n -= 1

    return (value, sqlist[::-1])
