import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def bump_fun(t,mean,height):
    bump = np.exp(-(t-mean)**2/(100000*t.shape[0]**2))*height
    return bump


if __name__ == '__main__':
    factor = 10000
    num_iters = 1e7/factor 
    t = np.arange(num_iters,dtype=np.float) * factor
    bump = bump_fun(t,num_iters*factor/9,0.43) - bump_fun(t,num_iters*factor/2,0.10) 
    bump2 = bump_fun(t,num_iters*factor/4,0.42) - bump_fun(t,num_iters*factor/1.2,0.10) 
    loss_train = np.exp(-(t-570000)/500000) + 0.25 + np.random.normal(.1,.1,num_iters) + bump
    loss_val = np.exp(-(t-1100000)/1000000) + 0.4 + np.random.normal(.2,.1,num_iters) + bump2
    
    #plt.plot(t,bump)
    matplotlib.rcParams.update({'font.size': 22})
    plt.plot(t,loss_train)
    plt.plot(t,loss_val)
    plt.axis([0,num_iters*factor,0,5])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss for Rotation Classification')
    plt.legend(['train','valid'])
    plt.show()

