import streamlit as st
import cv2
from PIL import Image
import numpy as np

def grab_cut():
        first = st.file_uploader("Upload 1st Image for grabcut", type=['jpg', 'png', 'jpeg'])
        second = st.file_uploader("Upload 2nd Image for grabcut", type=['jpg', 'png', 'jpeg'])
        if not first or not second:
               return None


        original_image_first = Image.open(first)
        original_image_first = np.array(original_image_first)

        st.image(original_image_first, caption="★ Original Image★")

        st.text("______________________________________________________________________________________________")
        original_image_second = Image.open(second)
        original_image_second = np.array(original_image_second)

        st.image(original_image_second, caption="★Original Image★")
        mask = np.zeros(original_image_first.shape[:2],np.uint8) 

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = (300,120,470,350)
        # this modifies mask 
        cv2.grabCut(original_image_first,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # adding additional dimension for rgb to the mask, by default it gets 1
        # multiply it with input image to get the segmented image
        img_cut = original_image_first*mask[:,:,np.newaxis]

        #plt.subplot(211),plt.imshow(img)
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(212),plt.imshow(img_cut)
        #plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
        #plt.show()
        #label = "✵ Origin image✵" 
        #st.image(original_image_first, caption=label)
        label = "✵ output image✵" 
        st.image(img_cut, caption=label)



        #original_image = cv2.imread('cartoon.jpg')
        mask = np.zeros(original_image_second.shape[:2], np.uint8)
        backgroundModel = np.zeros((1, 65), np.float64)
        foregroundModel = np.zeros((1, 65), np.float64)
        rectangle = (20, 100, 150, 150)
        cv2.grabCut(original_image_second, mask, rectangle,backgroundModel, foregroundModel,3, cv2.GC_INIT_WITH_RECT)



        mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')


        
        image = original_image_second * mask2[:, :, np.newaxis]

        #label = "✵ Origin image✵" 
        #st.image(original_image_second, caption=label)
        label = "✵ output image✵" 
        st.image(image, caption=label)
        #plt.subplot(211),plt.imshow(original_image)
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(212),plt.imshow(image)

        #plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
        #plt.show()
    
    
def main_opration():
    st.title("Grabcut")
    grab_cut()
    st.text("_____________________________________________________________________________________________________________")

if __name__ == "__main__":
    main_opration()
