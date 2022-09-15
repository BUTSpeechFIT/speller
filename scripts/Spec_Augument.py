#!/usrt/bin/bash


import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.pyplot.viridis()




#batch_size=20
#time=215

#inp=np.random.random((batch_size,time,249))

#----------------------------------------------------------------
def Spec_Aug(inp,min_F_bands,max_F_bands,time_drop_max,time_window_max):
        
        time_drop_percentage=time_drop_max
        max_freq_drop_percentage=max_F_bands
        time_drop_window=time_window_max

        time_drop_window=int(time_drop_window)
        #print(time_drop_window)
        feat=np.ones_like(inp,dtype=np.float32)
        time=feat.shape[1]
        #--------------------------------------------
        #---------------------------------------------
        #no_of_freq_masks=int(np.random.randint(max_freq_drop_percentage,size=1))
        no_of_freq_masks=max_freq_drop_percentage
        for _ in range(no_of_freq_masks):
                freq_mask=int(np.random.randint(inp.shape[2],size=1))
                #print("freq_mask",freq_mask)
                feat[:,:,freq_mask]=0
        #------------------------------------------------
        #------------------------------------------------
        #time_drop_percentage=2
        time_drop_masks=int(time*time_drop_percentage/100)
        #-----------------------------------------------
        for _ in range(time_drop_masks):
            time_mask=int(np.random.randint(time-5-time_drop_window,size=1))
            #print("time_ask",time_mask)
            feat[:,time_mask:time_mask+time_drop_window,:]=0
        #----------------------------------------------
        return feat
#-------------------------------------------------------------------

#----------------------------------------------------------------
def Spec_Aug_freqCont(inp,min_F_bands,max_F_bands,time_drop_max,time_window_max):
        """both time and frequency drops are continous,features are mean variance normalized"""
        
        F = int(max_F_bands)
        T = int(time_window_max)     
        time_drop_percentage = time_drop_max        
        
        feat = np.ones_like(inp, dtype=np.float32)
        time = feat.shape[1]

        
        ###sample freq masks from uniform distribution 
        f  = int(np.random.randint(low=1, high=F, size=1))
        f0 = int(np.random.randint(inp.shape[2]-(f-1), size=1))
        feat[:, :, f0:f0+f] = 0
 
        
        ##get number of time masks        
        time_drop_masks = int( time*time_drop_percentage/100 )
        
        for _ in range(time_drop_masks):
            t = int(np.random.randint(low=1, high=T, size=1))
            t0 = int(np.random.randint((time-5)-t, size=1))
            feat[:, t0:t0+t, :]=0
        
        return feat
#-------------------------------------------------------------------


"""
time_drop_percentage=2
max_freq_drop_percentage=10
time_drop_window=2
#import pdb; pdb.set_trace()

for i in range(10):
        batch_size=20
        time=100+np.random.randint(low=2,high=5,size=1)[0]
        inp=np.random.random((batch_size,time,249))
        
        #time_drop_percentage = np.random.randint(low=2,high=5,size=1)[0]
        #max_freq_drop_percentage = np.random.randint(low=30,high=80,size=1)[0]
        #time_drop_window = np.random.randint(low=1,high=4,size=1)[0]

        #min_F_bands=30;max_F_bands=80;time_drop_max=5;time_window_max=40;
        #spec_aug_mask=Spec_Aug_freqCont(inp,min_F_bands,max_F_bands,time_drop_max,time_window_max)

        print("time_drop_percentage,max_freq_drop_percentage,time_drop_window",time_drop_percentage,max_freq_drop_percentage,time_drop_window)
        spec_aug_mask=Spec_Aug(inp,time_drop_percentage,max_freq_drop_percentage,time_drop_window)
        print(inp.shape)




name='__fig____spec_augunent'
#import pdb;pdb.set_trace()
fig = plt.figure()
fig, ax = plt.subplots(1)
#plt.subplot(110)
#ax[0].imshow(inp[0].T,origin='lower')

#plt.subplot(111)
#ax[0].imshow((inp[0]*spec_aug_mask[0]).T,origin='lower')

ax.imshow((inp[0]*spec_aug_mask[0]).T,origin='lower')
#plt.subplot(2, 2, 3)
#plt.imshow(feat3[0].T,origin='lower')
#plt.subplot(2, 2, 4)
#plt.imshow(II.T,origin='lower')
plt.savefig(name,Bbox='tight', orientation='landscape')
plt.close()
#plt.show()

"""















