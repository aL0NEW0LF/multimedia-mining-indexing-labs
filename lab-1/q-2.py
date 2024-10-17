import cv2

# Approach 

nomimage = "assets/fondvert.png"

img = cv2.imread(nomimage)

cv2.imshow('1', img)

## Mask de base
### Conversion de l'image dans l'espace LAB 
### Seuil du canal a pour isoler le fond vert 
### Masquage de l'image originale avec un masque binaire

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
a_channel = lab[:,:,1]
th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
masked = cv2.bitwise_and(img, img, mask = th)
m1 = masked.copy()
m1[th==0]=(255,255,255)

cv2.imshow('1.5', m1)

## Supprimer l'ombre verte le long de la bordure
### Conversion de l'espace LAB de l'image masquée 
### Normalisation du canal a mlab[ :,:,1] pour utiliser toute la plage d'intensité entre [0-255] 
### Seuil binaire inverse pour sélectionner la zone avec des bordures vertes

mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
dst = cv2.normalize(mlab[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('2', dst)

threshold_value = 100
dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow('3', dst_th)

mlab2 = mlab.copy()
mlab[:,:,1][dst_th == 255] = 127

cv2.imshow('4', mlab2)

img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
img2[th==0]=(255,255,255)

# Affichage de l'image originale
cv2.imshow('5', img2)

# Attendre une touche
cv2.waitKey(0)

# Libérer les fenêtres 
cv2.destroyAllWindows()