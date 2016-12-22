
# coding: utf-8

# In[3]:

#image transformation test
get_ipython().magic('run image_transformation # include functions to test for')

#four source coordinates
src = np.float32(
    [[92,70],
     [246,16],
     [82,104],
     [269,38]])
    
#     four desired coordinates
dst = np.float32(
    [[80,60],
    [250,60],
    [80,100],
     [250,100]])
img = mpimg.imread('stopsign.jpg')
warped_img = warp(img,src,dst,0)
plotImageSet([img,warped_img])
warped_back = warp(warped_img,src,dst,1)
plotImageSet([img,warped_img,warped_back])


# In[ ]:



