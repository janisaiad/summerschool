# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Loading data
x = np.load('x.npy')

Nx = x.shape[0]

f_train = np.load('f_train_data.npy')
f_test  = np.load('f_test_data.npy')

Ntrain = f_train.shape[0]
Ntest  = f_test.shape[0]

u_test_type1  = np.load('u_test_data_c0.2.npy')
u_test_type2  = np.load('u_test_data_c0.5.npy')
u_test_type3  = np.load('u_test_data_c1.0.npy')

# %%

# Plotting type 1 samples
fig,axs = plt.subplots(5,4,figsize=(20,25))
axs = axs.flatten()
for s in range (5):
    axs[4*s].plot(x,f_test[s])
    axs[4*s].set_xlabel('x')
    axs[4*s].set_ylabel('f')
    axs[4*s].set_title(f'Samples {s+1}')
        
    axs[4*s+1].plot(x,u_test_type1[s])
    axs[4*s+1].set_xlabel('x')
    axs[4*s+1].set_ylabel('u')  
    axs[4*s+1].set_title(f'c=0.2')

    axs[4*s+2].plot(x,u_test_type2[s])
    axs[4*s+2].set_xlabel('x')
    axs[4*s+2].set_ylabel('u')  
    axs[4*s+2].set_title(f'c=0.5')

    axs[4*s+3].plot(x,u_test_type3[s])
    axs[4*s+3].set_xlabel('x')
    axs[4*s+3].set_ylabel('u')  
    axs[4*s+3].set_title(f'c=1.0')

