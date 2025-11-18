import numpy as np

from Actfunc import*
from init import*
from Loss import*

class Layer:

    def __init__(self,dim_in,dim_out,init_fn,act_fn=Relu):
        self.weights=init_fn(dim_in,dim_out)
        self.bias=np.reshape(init_fn(1,dim_out),-1)
        self.act_fn=act_fn


    def fw(self,input):
        output=self.weights@input + self.bias[:,None]
        Z=output
        if self.act_fn:
            output=self.act_fn(Z)
        return output,Z

        

    def gradient_last(self,onehot,A,Z):
        S=softmax_matrix(Z[-1])
        delta=S-onehot
        d_WL=np.einsum('ij,sj->jis',delta,A[-2]) # i stedet for d_WL=np.outer(delta,A[-2])
        d_BL=delta
        return d_WL, d_BL
    
class FFNN:

    def __init__(self,input_size,hidden_sizes,output_size, init_fn,act_fn):
        self.act_fn=act_fn
        self.layers=[]
        self.layers.append(Layer(dim_in=input_size,dim_out=hidden_sizes[0],init_fn=init_fn,act_fn=act_fn))
        for i in range(1,len(hidden_sizes)):
            self.layers.append(Layer(dim_in=hidden_sizes[i-1],dim_out=hidden_sizes[i],init_fn=init_fn,act_fn=act_fn))
        self.layers.append(Layer(dim_in=hidden_sizes[-1],dim_out=output_size,init_fn=init_fn,act_fn=False))

    

    def forward(self,input):

        A=[]
        Z=[]
        output=input
        for layer in self.layers:
            a,z=layer.fw(output)
            A.append(a)
            Z.append(z)
            output=a
        return output,A,Z
    
    def full_gradient(self,A,Z,onehot,input):
        gradients_w=[]
        gradients_b=[]
        dLi_dW,dLi_dB=self.layers[-1].gradient_last(onehot,A,Z)
        dLi_df=dLi_dB
        gradients_w.insert(0,np.sum(dLi_dW,axis=0))
        gradients_b.insert(0,np.sum(dLi_dB,axis=1))
        for i in range(len(self.layers)-2,0,-1):
            dLi_df=((self.layers[i+1].weights.T@dLi_df)*(self.act_fn(Z[i],derivative=True))) #
            dLi_dB=dLi_df #
            dLi_dW=np.einsum('ij,sj->jis',dLi_df,A[i-1]) # instead of dLi_dW=np.outer(dLi_df,A[i-1])
            gradients_w.insert(0,np.sum(dLi_dW,axis=0))
            gradients_b.insert(0,np.sum(dLi_dB,axis=1))
        dLi_df=((self.layers[1].weights.T@dLi_df)*(Z[0]>0)) #
        dLi_dB=dLi_df
        dLi_dW=np.einsum('ij,sj->jis',dLi_df,input)# instead of dLi_dW=np.outer(dLi_df,input)
        gradients_w.insert(0,np.sum(dLi_dW,axis=0))
        gradients_b.insert(0,np.sum(dLi_dB,axis=1))
        return gradients_w,gradients_b
    
