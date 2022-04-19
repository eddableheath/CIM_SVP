import subprocess
import os
import shutil
import time
import numpy as np
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

    Jij = np.zeros((N, N))
    
    for idx in range(1, len(file[:, 0])):
        i, j, val = file[idx, :]
        Jij[i-1, j-1] = Jij[j-1, i-1] = val
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

alpha_i = 1.0/np.sqrt(np.sum(Jij**2, axis=0))

print(alpha)

tamp_start = 1.0
tamp_raise = 1.5

beta = 0.3


def E(mu, a):
    return np.sum(mu*mu/4.0 - a*mu*mu/2.0 + alpha*mu*np.dot(Jij, mu))


def E_ising(mu):
    return np.sum(np.sign(mu)*np.dot(Jij, np.sign(mu)))


def limit_range(mu, tamp):
    return np.minimum(np.maximum(mu, -np.sqrt(tamp)*1.5), np.sqrt(tamp)*1.5)


def step(mu, ei, dt, a, tamp):
    fi = alpha*np.dot(Jij, mu)*ei
    dmu = -mu**3 + a*mu - fi
    
    dei = -beta*ei*(mu**2 - tamp) #CAC
    #dei = -beta*ei*(fi**2 - tamp) #CFC
    
    return limit_range(mu + dmu*dt, tamp), ei + dei*dt


print( "ones : ", E_ising(np.ones(N)))

mu = np.zeros(N)


dt = 0.025
a = -2.0

nt = 2000

#NOTE: a = p - 1

a_raise = 2.0
a_end = a + a_raise
a_start = a


def traj(mu, ei):
    E_opt = 0
    for i in range(0,nt):
        a = a_start  + (float(i)/nt)*a_raise
        tamp = tamp_start + (float(i)/nt)*tamp_raise
        mu, ei = step(mu, ei, dt, a, tamp)
        E_opt = min(E_ising(mu), E_opt)
    
    return mu, ei, E_opt


num_rep = 100
num_success = 0

E_s = []

for i in range(num_rep):
    mu = np.random.rand(N)-0.5
    ei = 10*np.ones(N)
    mu, ei, E_opt = traj(mu, ei)
    #print "mu: ", mu
    #print np.dot(Jij, mu)
    print("")
    print("E opt " + str(E_opt - ground_E))
    print("E " + str(E_ising(mu)))
    print("")
    E_s.append(E_opt)
    
    if(abs(E_opt - ground_E) < 0.00000001):
        num_success += 1

min_found = np.min(E_s)

print("success rate " + str(float(num_success)/num_rep))
print("")
print("min found: " + str(min_found))
print("freq: " + str(float(np.sum(np.array(E_s) == min_found))/num_rep))


#plot trajectoreis:
def plot_traj(mu, ei, nt):
    stored = np.zeros((N,nt))
    stored_e = np.zeros((N,nt))
    stored_ising_E = np.zeros((nt))
    for i in range(0,nt):
        a = a_start  + (float(i)/nt)*a_raise
        tamp = tamp_start + (float(i)/nt)*tamp_raise
        mu, ei = step(mu, ei, dt, a, tamp)
        stored[:, i] = mu
        stored_e[:, i] = ei
        stored_ising_E[i] = E_ising(mu)
    
    plt.title("mu traj")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((-2.5,2.5))
    
    for i in range(N):
        plt.plot(x_axis, stored[i, :])
    
    plt.show()
    
    plt.close()
    
    plt.title("ei traj")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((0,5.0))
    
    for i in range(N):
        plt.plot(x_axis, stored_e[i, :])
    
    plt.show()
    
    plt.close()
    
    plt.title("resid Ising E traj")
    
    x_axis = np.array(range(nt))*dt
    
    plt.ylim((0,10.0))
    
    E_ref = ground_E
    if(E_ref > 0):
        E_ref = min_found
        
    print(np.min(stored_ising_E))
    
    plt.plot(x_axis, stored_ising_E - E_ref)
    
    plt.show()

mu = np.random.rand(N)-0.5
ei = 10*np.ones(N)
plot_traj(mu, ei, 200*100)

mu = np.random.rand(N)-0.5
ei = 10*np.ones(N)
plot_traj(mu, ei, 200*100)

mu = np.random.rand(N)-0.5
ei = 10*np.ones(N)
plot_traj(mu, ei, 200*100)






