import numpy as np
import matplotlib.pyplot as plt


def QThist(x,y, N=5, thresh=4, rng=[], density=False):
    
    # start w/ 2x2 array of False leafs
    Mnext = np.empty((2**1,2**1),dtype='Bool')*False
    
    # the 5 quantities to save in our Tree
    num = np.array([])
    xmin = np.array([])
    xmax = np.array([])
    ymin = np.array([])
    ymax = np.array([])
    
    # Step thru each level of the Tree
    for k in range(1, N+1):
        if len(rng) == 0:
            dx = (np.nanmax(x) - np.nanmin(x)) / (2**k)
            dy = (np.nanmax(y) - np.nanmin(y)) / (2**k)
            rng = [[np.nanmin(x)-dx/4, np.nanmax(x)+dx/4], 
                   [np.nanmin(y)-dy/4, np.nanmax(y)+dy/4]]

        # lazily compute histogram of data at this level
        H1, xedges1, yedges1 = np.histogram2d(x, y, range=rng, bins=2**k,)

        # any leafs at this level to pick, but NOT previously picked?
        if k<N:
            M1 = (H1 <= thresh)
        if k==N:
            # unless we on the last level, then pick the rest of the leafs
            M1 = ~Mnext

        Mprep = np.empty((2**(k+1),2**(k+1)),dtype='Bool')*False

        # check leafs at this level
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                # up-scale the leaf-picking True/False to next level
                if k<N:
                    Mprep[(i*2):((i+1)*2),(j*2):((j+1)*2)] = M1[i,j] | Mnext[i,j]

                # if newly ready to pick, save 5 values
                if M1[i,j] & ~Mnext[i,j]:
                    num = np.append(num, H1[i,j])
                    xmin = np.append(xmin, xedges1[i])
                    xmax = np.append(xmax, xedges1[i+1])
                    ymin = np.append(ymin, yedges1[j])
                    ymax = np.append(ymax, yedges1[j+1])

        Mnext = Mprep

    if density:
#   following example from np.histogram:
#   result is the value of the probability *density* function at the bin, 
#   normalized such that the *integral* over the range is 1
        num = num / ((ymax - ymin) * (xmax - xmin)) / num.sum()
        
    return num, xmin, xmax, ymin, ymax


def QTcount(x,y,xmin, xmax, ymin, ymax, density=False):
    '''
    given rectangular output ranges for cells/leafs from QThist
    count the occurence rate of NEW data in these cells
    '''
    
    num = np.zeros_like(xmin)
    for k in range(len(xmin)):
        num[k] = np.sum((x >= xmin[k]) & (x < xmax[k]) & 
                        (y >= ymin[k]) & (y < ymax[k]))
        
    
    if density:
#   following example from np.histogram:
#   result is the value of the probability *density* function at the bin, 
#   normalized such that the *integral* over the range is 1
        num = num / ((ymax - ymin) * (xmax - xmin)) / num.sum()

    return num


def QThistPlot(x,y,num, xmin, xmax, ymin, ymax):
    fig = plt.figure(figsize=(7,8))
    ax = fig.add_subplot(111)
    plt.scatter(x,y, c='w', s=5, alpha=0.5)
    for l in range(len(num)):
        ax.add_patch(plt.Rectangle((xmin[l], ymin[l]), xmax[l]-xmin[l], ymax[l]-ymin[l], 
                                   fc ='none', ec='C0', lw=1, alpha=0.5))

    plt.gca().invert_yaxis()

    plt.xlabel('$G_{BP} - G_{RP}$ (mag)')
    plt.ylabel('$M_G$ (mag)')
    plt.title('Depth='+str(k)+'/'+str(N))
    