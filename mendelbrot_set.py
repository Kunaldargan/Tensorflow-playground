from IPython.display import Image, display
import io
import PIL.Image

import tensorflow as tf
import numpy as np

#MENDELBROT SET using TF


def to_uint8( data ) :
    # maximum pixel
    latch = np.zeros_like( data )
    latch[:] = 255
    # minimum pixel
    zeros = np.zeros_like( data )

    # unrolled to illustrate steps
    d = np.maximum( zeros, data )
    d = np.minimum( latch, d )

    # cast to uint8
    return np.asarray( d, dtype="uint8" )

def DisplayFractal(img,a, fmt='jpeg'):
    
        "Display colourful set"
        #acyclic=None
        a_cyclic=(6.28*a/20.0).reshape(list(a.shape)+[1])
        #print(type(a_cyclic))
        temp1 = np.array([10+20*np.cos(a_cyclic),30+50*np.sin(a_cyclic),155-80*np.cos(a_cyclic)])
        temp2 = np.array([10+20*np.cos(a_cyclic),30+50*np.sin(a_cyclic),155-80*np.cos(a_cyclic)])
        img = [temp1,temp2]
        #print(img,type(img))
        temp3=int(a.max())
        #img=list(img)
        img[temp3]=0
        #img(temp3) = 0
    
        print(img)
        a=np.array(img)
        #print(a)
        a=np.uint8(np.matrix.clip(a,0,255))
        print(type(a))
        f=StringIO()
        print('hi')
        """
        b=np.uint8(np.matrix.clip(a[0],0,255))
        c=np.uint8(np.matrix.clip(a[1],0,255))
        """
        PIL.Image.fromarray(a).save(f, fmt)
        print(a)
        print(PIL.Image.fromarray(a).save(f, fmt))
        #PIL.Image.fromarray(c).save(f, fmt) 
        print(f.getvalue())
        
        display(displayImage(data=f.getvalue())) 
        
    
        print("type error ")



#Session and variables
sess= tf.InteractiveSession()

#NP to create 2D array of complex numbers

Y,X=np.mgrid[-1.3:1.3:0.005,-2:1:0.005]
Z=X+1j*Y

#tensors defined
xs=tf.constant(Z.astype(np.complex64))
zs=tf.Variable(xs)
ns=tf.Variable(tf.zeros_like(xs, tf.float32))

tf.global_variables_initializer().run()

#compute the values of :z^2 + x
zs_=zs*zs + xs

#divergence
not_diverged = tf.abs(zs_) < 4

#update zs and iteration

step= tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))
img=np.zeros(shape=(3,2,2))
for i in range(200) :
    step.run()
    DisplayFractal(img,ns.eval())
    
