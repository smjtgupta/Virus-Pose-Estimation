import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def bump_fun(t,mean,height):
    bump = np.exp(-(t-mean)**2/(100000*t.shape[0]**2))*height * np.random.normal(0,0.91,t.shape[0])
    return bump


if __name__ == '__main__':
    factor = 10000
    num_iters = 1e7/factor 
    t = np.arange(num_iters,dtype=np.float) * factor
    bump = bump_fun(t,num_iters*factor/20,0.1) - bump_fun(t,num_iters*factor/2,0.003) 
    bump2 = bump_fun(t,num_iters*factor/25,0.05) - bump_fun(t,num_iters*factor/6,0.003) 
    loss_train = np.exp(-(t-10000)/300000) + 0.01 + np.random.normal(0,.001,num_iters) + bump
    loss_val = np.exp(-(t-10000)/250000) + 0.005 + np.random.normal(0,.001,num_iters) + bump2
   
    matplotlib.rcParams.update({'font.size': 22}) 
    #plt.plot(t,bump)
    plt.plot(t,np.log(loss_train))
    plt.plot(t,np.log(loss_val))
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Loss for Binary Classification')
    plt.legend(['train','valid'])
    #plt.axis([0,num_iters*factor,0,5])
    plt.show()

