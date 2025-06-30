import numpy as np
from scipy import integrate
#Task difficulty $d$, current knowledge $k$, workload $w$ (integer value; \[1, 3] indicates normal workload, \[4, 6] indicates excessive workload), and drowsiness $s$ (discrete value, only 0 or 1).
def compute_k(k,w,d):
    if k>=d:
        k_next=d
    else:
        if w<=3:
            k_next=(1+np.exp(-w))*k
        else:
            k_next=(1-np.exp(w-7))*k
    return k_next

def integrand1(x,k):
    return 0.8*(k/x)*0.85**x*np.log(1/0.85)

def integrand2(x):
    return 0.8*0.85**x*np.log(1/0.85)

if __name__ == "__main__":
    k=1
    d=20
    #p = 0.1   # Probability that the difficulty value is infinite, ranging between (0, 1)
    #elta=0.84 # Exponential distribution hyperparameter, taking values (0, 1)
    #delta=1  # Expected reward function hyperparameter, taking values (0, infinity)
    b_stack=[]
    
    W=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/work_h4.csv', delimiter=',')
    for i in range(122):
        fenzi1,error1=integrate.quad(integrand1,k,np.inf,args=(k,))
        fenzi=fenzi1+0.2
        fenmu1,error2=integrate.quad(integrand2,k,np.inf)
        fenmu=fenmu1+0.2
        b_h=fenzi/fenmu
        b_stack.append(b_h)
        #print(k)
        #print(b_h)
        w=W[i]
        k_next=compute_k(k,w,d)
        k=k_next
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/b_h4.csv', b_stack, delimiter=',')     
        