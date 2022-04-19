import subprocess
import os
import shutil
import time
import numpy as  np

import random

import matplotlib.pyplot as plt


instance_set_path = "../InstanceSets/BENCHSKL/"

instance_file_dir = "BENCHSKL_800_100/"
inst_file_common_name = "BENCHSKL_800_100_"

def read_Jij(inst_num, N):
    
    path = instance_set_path + instance_file_dir + inst_file_common_name + str(inst_num+1)
    
    file = np.loadtxt(path)
    
    
    Jij = np.zeros((N,N))
    
    for idx in range(len(file[:,0])):
        i,j,val = file[idx,:]
        Jij[i-1,j-1] = Jij[j-1,i-1] = val
    return Jij
    
def read_ground_E(inst_num, N):
    
    path = instance_set_path + instance_file_dir + inst_file_common_name + "SOL"
    
    file = np.loadtxt(path)
    
    Jij = read_Jij(inst_num, N)
    
    #print file[inst_num]
    #print np.sum(Jij)
    
    return -np.sum(Jij)*0.5 - 2*file[inst_num]
    


instance_set_path_gset = "../InstanceSets/Gset/"


instance_file_dir_gset = "Gset_2000_v1/"
inst_file_common_name_gset = "G"


def read_Jij_gset(inst_num, N):
    
    path = instance_set_path_gset + instance_file_dir_gset + inst_file_common_name_gset + str(inst_num+1) + ".txt"
    
    file = np.loadtxt(path)
    
    
    
    Jij = np.zeros((N,N))
    
    for idx in range(1,len(file[:,0])):
        i,j,val = file[idx,:]
        Jij[i-1,j-1] = Jij[j-1,i-1] = val
    return Jij

def read_ground_E_gset(inst_num, N):
    
    path = instance_set_path_gset + instance_file_dir_gset + "SOL.txt"
    
    file = np.loadtxt(path)
    
    Jij = read_Jij_gset(inst_num, N)
    
    #print file[inst_num]
    #print np.sum(Jij)
    
    return np.sum(Jij)*0.5 - 2*file[inst_num]

def compute_ground_E(Jij):
    sol_base_vectors=np.zeros([pow(2,N),N])
    for idx in range(pow(2,N)):
        str_bin = format(idx,"0"+str(N)+"b")
        for jdx in range(N):
            sol_base_vectors[idx][jdx] = 1-2*int(str_bin[jdx]) 
    
    
    miNE = N*N
    E_1 = N*N
    miNsol = []
    
    for sol in sol_base_vectors:
        E = np.dot(np.dot(Jij, sol), sol)
        if(E <= miNE+0.000000001):
            miNE = E
            miNsol = sol
        else:
            if(E < E_1):
                E_1 = E
    return miNE

instance_file_dir = "BENCHSKL_100_100/"
inst_file_common_name = "BENCHSKL_100_100_"

N = 100;

Jij = np.zeros((N,N))



ground_E = 2*read_ground_E(33, N)

Jij = -read_Jij(33, N)


#ground_E = 2*read_ground_E_gset(21, N)

#Jij = read_Jij_gset(21, N)


print(Jij)

if(N<16):
    ground_E = compute_ground_E(Jij)
    print("ground E " + str(ground_E))

Jij_sum = 0.0
for i in range(0,N):
    for j in range(0,i):
        Jij_sum +=  abs(Jij[i,j])

Jij_sum = np.sqrt(Jij_sum/N)

alpha = 1.0/Jij_sum

alpha_i = 1.0/np.sqrt(np.sum(Jij**2, axis = 0))

print(alpha)

tamp = 1.0
tamp_raise = 0.0

beta  = 0.0
beta_init = 0.3
beta_raise = -0.2




def E_ising(mu):
    return np.sum(np.sign(mu)*np.dot(Jij, np.sign(mu)))
    
def limit_range(x, tamp):
    
    k = 200.0
    return np.maximum(np.minimum(x, k), -k)



ei_max = 1000

def F(x):
    return 2*x/(1 + x**2)*1.0 + 0.05*x


# nonliennar feedback (set to tanh)
def nl(x):
    #return x
    return np.tanh(x)
    #return np.minimum(1.0,np.maximum(-1.0, x))
    
print(alpha_i)
print(alpha)
alpha = alpha


k = 0.2

#integration time step
def step(mu, ei, dt, a, tamp, beta, c):
    dt = dt*1.0
    
    
    fi = alpha*np.dot(Jij,(mu))
    #fi = nl3(fi, tamp)
    
    dmu = -1.0*mu**3 + a*mu - nl(fi)  -  k*(fi - ei)
    dei = -beta*(ei - fi)
    
    
    
    
    
    return   limit_range(mu + dmu*dt, tamp),  ei + dei*dt


mu = np.zeros(N)

c_init = 1.0
c_end = 3.0

Exp = 1

#NOTE: a = p -1

dt= 0.4
a = -2.0
a_raise = 2.0

nt = int(1000/dt)

#scheduling funcntion (for parameter modulation)
def a_func(factor):
    return  a + a_raise*factor**1.0


#scheduling funcntion (for parameter modulation)
def factor_func(factor):
    
    return factor**1.0




def traj(mu, ei):
    E_opt = 0
    for i in range(0,int(nt)):
        beta = beta_init + beta_raise*factor_func(float(i)/nt)
        c = c_init + (c_end - c_init)*factor_func(float(i)/nt)**Exp
        mu, ei = step(mu, ei, dt, a_func(factor_func(float(i)/nt)), tamp + tamp_raise*factor_func(float(i)/nt), beta, c)
        E_opt = min(E_ising(mu), E_opt)
    
    
    E_opt = min(E_ising(mu), E_opt)
    
    return mu, ei, E_opt


num_rep = 100
num_success = 0

E_s = []

best_E = 0


for i in range(num_rep):
    mu =  np.random.normal(0, 0.1, N)
    
    ei = np.zeros(N)
    
    mu ,ei, E_opt = traj(mu, ei)
    
    #print "mu: ", mu
    #print np.dot(Jij, mu)
    
    print("")
    print("E opt " + str(E_opt - ground_E))
    print("E " + str(E_ising(mu)))
    print("")
    
    
 
    E_s.append(E_opt)
    
    if(abs(E_opt - ground_E) < 0.00000001):
        num_success += 1

E_s = np.array(E_s)
min_found = np.min(E_s)



print("success rate " + str(float(num_success)/num_rep))

print("")
print("min found: " + str(min_found))
print("freq: " + str(float(np.sum(np.array(E_s) == min_found))/num_rep))
print("avg E resid: "  + str(np.average(E_s - ground_E)))



#plot trajectoreis:


def plot_mu_correlation(stored):
    
    fig, ax = plt.subplots(2,2)
    
    
    ax[0,0].scatter(stored[:, 0], stored[:, 5-1], alpha = 0.5)
    ax[0,0].set_title("5 time steps")
    ax[0,1].scatter(stored[:, 0], stored[:, 25-1], alpha = 0.5)
    ax[0,1].set_title("10 time steps")
    ax[1,0].scatter(stored[:, 0], stored[:, 60-1], alpha = 0.5)
    ax[1,0].set_title("40 time steps")
    ax[1,1].scatter(stored[:, 0], stored[:, 100-1], alpha = 0.5)
    ax[1,1].set_title("100 time steps")
    
    plt.show()    

def plot_traj(mu, ei, nt):
    stored = np.zeros((N,nt))
    stored_e = np.zeros((N,nt))
    stored_ising_E = np.zeros((nt))
    for i in range(0,nt):
        beta = beta_init + beta_raise*factor_func(float(i)/nt)
        c = c_init + (c_end - c_init)*factor_func(float(i)/nt)**Exp
        mu , ei= step(mu, ei, dt, a_func(factor_func(float(i)/nt)), tamp + tamp_raise*factor_func(float(i)/nt), beta, c)
        stored[:, i] = mu
        stored_e[:, i] = ei
        stored_ising_E[i] = E_ising(mu)
    
    #plot_mu_correlation(stored)
    
    plt.title("mu traj")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((-2.0,2.0))
    
    for i in range(N):
        plt.plot(x_axis, stored[i, :])
    
    plt.show()
    
    plt.close()
    
    plt.title("mu traj 10/800")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((-1.5,1.5))
    
    for i in range(10):
        plt.plot(x_axis, stored[i, :])
    
    plt.show()
    
    plt.close()
    
    plt.title("ei traj")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((-5,20))
    
    for i in range(N):
        plt.plot(x_axis, stored_e[i, :])
    
    plt.show()
    
    plt.close()
    
    plt.title("resid Ising E traj")
    
    x_axis = np.array(range(nt))*dt
    
    #plt.ylim((0,20.0))
    
    E_ref = ground_E
    if(E_ref > 0):
        E_ref = min_found
        
    print(np.min(stored_ising_E))
    
    plt.plot(x_axis, stored_ising_E - E_ref)
    
    plt.show()
    
    plt.title("resid Ising E traj log")
    
    x_axis = np.array(range(nt))*dt
    
    #plt.ylim((0,20.0))
    plt.yscale("symlog")
    
    E_ref = ground_E
    if(E_ref > 0):
        E_ref = min_found
        
    print(np.min(stored_ising_E))
    
    plt.plot(x_axis, stored_ising_E - E_ref)
    
    plt.show()



print("ground E ", ground_E)

mu = np.random.normal(0, 0.1, N)
ei = np.zeros(N)
plot_traj(mu, ei, 10*10)

mu = (np.random.rand(N)-0.5)*0.1
ei = np.zeros(N)
plot_traj(mu, ei, nt)

mu = (np.random.rand(N)-0.5)*0.1
ei = np.zeros(N)
plot_traj(mu, ei, nt)

mu = (np.random.rand(N)-0.5)*0.1
ei = np.zeros(N)
plot_traj(mu, ei, nt)

mu = np.random.rand(N)-0.5
ei = np.zeros(N)
plot_traj(mu, ei, 200*100)




