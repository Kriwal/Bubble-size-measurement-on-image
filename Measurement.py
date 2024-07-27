#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:53:30 2021

@author: krishanbumma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage as ski
from skimage import io,color,measure,util
from skimage.filters import difference_of_gaussians,threshold_local
from scipy import ndimage
from skimage.draw import circle_perimeter, disk
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed
import pandas as pd




# code de mesure de taille de bulles
# permet de mesurer la taille des bulles sur une image du
# type bulles monocouche de bulles sur un liquide
# l'algorithme suit les étapes suivantes : 
#   -Ouverture de l'image
#   -filtre pass bande pour mieux détecter les bords des bulles (sans bruit aux petites échelles) et éliminer le vignétage (grandes échelles)
#   -threshold 
#   -mesure des zones blanches circulaires (type analyse particles sur imageJ)
#   -supression des zones intérieures et passage en négatif / les zones représentées par les bulles sont désormais noires
#   -repétitions de : 
#       -calcul de la distance de chanque pixel blanc au pixel noir le plus proche
#       -algorithmeme de watershed 
#       -mesure plus élimination des zone circulaires
#   -dessin des zones circulaires sur une copie de la photo initiale 
#   -enregistrement des données 


# Plus l'image est nette et sans flou de mouvement, meilleure sera la détection. 
# Toutefois, les différents filtres permettent de rattraper certaines images.

# le code prend en entrée une image   (filename), et forme une liste des rayons des bulles mesurées
# et crée une image ont sont affichées les bulles mesurées 



filename = 'Example_image.tif' # chemin de l'image à traiter

#  PARAMETRES MODIFIABLES
NumOfErosions=8       #Permet d'éliminer des irrécularités empêchant la detection, mais rend tout rond.
FillingHoles=False         
center_filling_ratio = 1.3   #entre 1 et 2 : 2 = rempli les trous avec un cercles de la taille exacte du trou. 1 = Un cercle 2 fois plus grand. Tres efficace pour enlever les petites erreurs dans le threshold à cause de reflections etc..
r_min = 20   #rayon minimum mesuré (en pixels)
minimum_circularity = 0.81

# On laisse le scale 1 ici, on y touche plus tard dans la méthode, une fois sur numbers
pixels_to_um=1

bpfilterlow= 3 # 3-5 pour lisser les reflections internes
bpfilterhigh= 400  #200-600 usually    600 = enleve uniquement l'inhomogeneite de luminosité. 250 = rend tout un peut flou. 



orig_image=io.imread(filename, plugin='pil')
image=io.imread(filename, plugin='pil', as_gray=True)



#Transforme orig_image en image grise RVB pour la création du jpg
for i in range(orig_image.shape[0]):
    for j in range(orig_image.shape[1]):
        Norm=min(image[i,j]*255*1,255)
        orig_image[i,j]=[Norm,Norm,Norm,255]
    



# plt.imshow(image)
# plt.title('original image')
# plt.show()

   
filtered_image = difference_of_gaussians(image,bpfilterlow,bpfilterhigh, mode='reflect')
 


# plt.imshow(filtered_image)
# plt.title('bandpass filter')
# plt.show()


kernel=None #np.ones((3,3))


# Peut aider pour eliminer le bruit due au bandpass filter
filtered_image=ski.morphology.dilation(filtered_image, footprint=kernel)
filtered_image=ski.morphology.dilation(filtered_image, footprint=kernel)
filtered_image=ski.morphology.dilation(filtered_image, footprint=kernel)

filtered_image=ski.morphology.erosion(filtered_image, footprint=kernel)
filtered_image=ski.morphology.erosion(filtered_image, footprint=kernel)
filtered_image=ski.morphology.erosion(filtered_image, footprint=kernel)


block_size = 61
local_thresh = threshold_local(filtered_image, block_size, offset=0)
binary_local = filtered_image > local_thresh

threshold_global_otsu = ski.filters.threshold_otsu(filtered_image)
global_otsu = filtered_image >= threshold_global_otsu

global_otsu = filtered_image >= (threshold_global_otsu+local_thresh)/2    #homemade treshold



# plt.imshow(np.uint8(global_otsu), cmap='gray')
# plt.title('thresholded')
# plt.show()
# plt.imsave(filename[:-4]+"Thresh.png",global_otsu,cmap='gray')

#global_otsu=ski.morphology.erosion(global_otsu)


"""
radius = 600

footprint = disk(radius)
local_otsu = ski.filters.rank.otsu(img, footprint)

global_otsu= img>= local_otsu/255
"""


mask = global_otsu == 255  #Sets TRUE for all 255 valued pixels and FALSE for 0
#print(mask)   #Just to confirm the image is not inverted. 


mask = clear_border(global_otsu)   #Removes edge touching grains. 

s = [[1,1,1],[1,1,1],[1,1,1]]
#label_im, nb_labels = ndimage.label(mask)
labeled_mask, num_labels = ndi.label(mask, structure=s)

#The function outputs a new image that contains a different integer label 
#for each object, and also the number of objects found.

# table is a dictionary mapping column names to data columns
# (NumPy arrays)
table = measure.regionprops_table(
    labeled_mask,
    properties=('label','area',
            'equivalent_diameter', 
            'orientation', 
            'major_axis_length',
            'minor_axis_length',
            'perimeter'),)
    

# Circularity test
condition = (table['minor_axis_length']/table['major_axis_length'] >minimum_circularity) & (table['area']*1.1>np.pi*(table['equivalent_diameter']**2/4)) & (table['area']/(table['major_axis_length']**2*np.pi/4)>minimum_circularity) & (table['area']/(table['minor_axis_length']**2*np.pi/4)<1.2)

# zero out labels not meeting condition
input_labels = table['label']
output_labels = input_labels * condition

filtered_lab_image = util.map_array(
  labeled_mask, input_labels, output_labels
)


#Let's color the labels to see the effect
img2 = color.label2rgb(filtered_lab_image, bg_label=0)

# plt.imshow(img2)
# plt.title('detected circular regions inside')
# plt.show()


clusters = measure.regionprops(filtered_lab_image, image)

propList = ['area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'major_axis_length',
            'centroid',
            'minor_axis_length',
            'perimeter']   

negative=np.ones(filtered_image.shape)-global_otsu


   
#First cell blank to leave room for header (column names)
#and cluster_props['area']/(cluster_props['major_axis_length']**2*np.pi/4)>0.9 and cluster_props['area']/(cluster_props['minor_axis_length']**2*np.pi/4)<1.1
    
for table in clusters:
    if table['major_axis_length']>0 and (table['minor_axis_length']/table['major_axis_length'] >minimum_circularity) & (table['area']*1.1>np.pi*(table['equivalent_diameter']**2/4)) & (table['area']/(table['major_axis_length']**2*np.pi/4)>minimum_circularity) & (table['area']/(table['minor_axis_length']**2*np.pi/4)<1.2):
        cluster_props=table
        r = cluster_props['major_axis_length']/center_filling_ratio
        center=cluster_props['centroid']
        rr, cc = disk(center, r, shape=negative.shape)
        negative[rr, cc] = 1
        


plt.imshow(negative,cmap='gray')
plt.title('Background after erosion')
plt.show()



distance = ndi.distance_transform_edt(negative)
# plt.imshow(np.power(distance,1),cmap='jet')
# plt.title('distance mapping')
# plt.show()

coords = peak_local_max(distance, footprint=np.ones((3, 3)))
mask = np.zeros(distance.shape, dtype=bool)

mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=negative)


    
Temp= color.label2rgb(labels, bg_label=0)
# plt.imshow(Temp)
# plt.title('watershed segmentation')
# plt.show()




clusters = measure.regionprops(labels, image)

propList = ['area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'major_axis_length',
            'minor_axis_length',
            'perimeter', 'centroid']   



#crée un fichier avec toutes les données mesurées

output_file = open(filename[:-4]+'_pix.csv', 'w')
output_file.write(',' + ",".join(propList) + ', , '+'\n') #join strings in array by commas, leave first cell blank
#First cell blank to leave room for header (column names)


NofCirc=0
selected=negative.copy()
NewNegative=negative.copy()    
for cluster_props in clusters:
    table=cluster_props
    if table['major_axis_length']>2*r_min and (table['minor_axis_length']/table['major_axis_length'] >minimum_circularity) & (table['area']*1.1>np.pi*(table['equivalent_diameter']**2/4)) & (table['area']/(table['major_axis_length']**2*np.pi/4)>minimum_circularity) & (table['area']/(table['minor_axis_length']**2*np.pi/4)<1.2) and table['minor_axis_length']>2*r_min:
        for prop in propList: 
            if(prop == 'area'): 
                to_print = cluster_props[prop]   #Convert pixel square to um square
            elif (prop=='equivalent_diameter'): 
                to_print = cluster_props[prop]
            elif(prop == 'orientation'): 
                to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
            else: 
                to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
            output_file.write(',' + str(to_print))
        output_file.write('\n')

        r = cluster_props['major_axis_length']/2
        center=cluster_props['centroid']
        rr, cc = disk(center, r, shape=negative.shape)
        x,y=circle_perimeter(np.uint(center[0]),np.uint(center[1]),np.uint(r),shape=negative.shape)
        selected[rr, cc] = 0.5
        NewNegative[rr, cc] = 0
        orig_image[x,y]=(255,0,0,255)   #trace des cercles rouge sur l'image grise RVB
        NofCirc+=1



# plt.imshow(selected,cmap='gray')
# plt.title('selected Ncircles ={}'.format(NofCirc))
# plt.show()

# df=pd.read_csv(filename[:-4]+'__TEST.csv')
# plt.hist(df["equivalent_diameter"]/2, bins=30)
# plt.title('equivalent_radius in pixels, number of circles = {}'.format(NofCirc))
# plt.show()

NewCircs=1
for i in range(NumOfErosions):
    if i==2 and FillingHoles==True : 
        NewNegative=ndimage.binary_fill_holes(NewNegative, structure=np.ones((5,5))).astype(int)

    if True : 
        
        
        negative=NewNegative
        
        for j in range(min(i,NumOfErosions)):
            negative=ski.morphology.dilation(negative,footprint=kernel)
        for j in range(min(i,NumOfErosions)):
            negative=ski.morphology.erosion(negative,footprint=kernel)
            
        
        distance = ndi.distance_transform_edt(negative)
            
        # plt.imshow(negative,cmap='gray')
        # plt.title('Background after erosion')
        # plt.show()

        # plt.imshow(np.power(distance,1),cmap='jet')
        # plt.title('distance mapping')
        # plt.show()
        
        coords = peak_local_max(distance, footprint=np.ones((3, 3)))
        mask = np.zeros(distance.shape, dtype=bool)
        
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=negative)
        
    
        # Temp= color.label2rgb(labels, bg_label=0)
        # plt.imshow(Temp)
        # plt.title('watershed segmentation')
        # plt.show()
        
    
        clusters = measure.regionprops(labels, image)
        

       
        #First cell blank to leave room for header (column names)
        selected=negative.copy()
        NewNegative=negative.copy()
        NewCircs=0    
        for cluster_props in clusters:
            table=cluster_props
            if table['major_axis_length']>2*r_min and (table['minor_axis_length']/table['major_axis_length'] >minimum_circularity) & (table['area']*1.1>np.pi*(table['equivalent_diameter']**2/4)) & (table['area']/(table['major_axis_length']**2*np.pi/4)>minimum_circularity) & (table['area']/(table['minor_axis_length']**2*np.pi/4)<1.2) and table['minor_axis_length']>2*r_min:
                
                for prop in propList: 
                    if(prop == 'area'): 
                        to_print = cluster_props[prop]   #Convert pixel square to um square
                    elif(prop == 'orientation'): 
                        to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
                    else: 
                        to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
                    output_file.write(',' + str(to_print))
                output_file.write('\n')
                
                r = cluster_props['major_axis_length']/2
                center=cluster_props['centroid']
                rr, cc = disk(center, r, shape=negative.shape)
                x,y=circle_perimeter(np.uint(center[0]),np.uint(center[1]),np.uint(r),shape=negative.shape)
                selected[rr, cc] = 0.5
                NewNegative[rr, cc] = 0
                orig_image[x,y]=(255,0,0,255)
                NewCircs+=1
                
        NofCirc+=NewCircs
        
    #     plt.imshow(selected,cmap='gray')
    #     plt.title('selected circles = {}'.format(NofCirc))
    # #  plt.savefig(filename[:-4]+'_selectedCircles.png', dpi=1000)
    #     plt.show()
    

plt.imshow(orig_image)
plt.title('selected circles = {}'.format(NofCirc))
plt.show()
io.imsave(filename[:-4] + '_selectedCircles.png', orig_image)


output_file.close()  #Closes the file, otherwise it would be read only. 


