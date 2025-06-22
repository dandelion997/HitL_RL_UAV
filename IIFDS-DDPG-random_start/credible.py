import numpy as np
from scipy import integrate
#任务难度d,当前知识k,工作负荷w(整数值，[1,3]负荷属于正常值，[4,6]负荷过重),是否犯困s(离散值，只取0或者1)
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
    #p=0.1   #难度值为无穷的概率,取值（0，1）
    #elta=0.84 #指数分布的超参数,取值（0，1）
    #delta=1  #预期报酬函数的超参数,取值（0，无穷）
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
        