import cv2
import numpy as np
import os

def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    '''
    Fonction pour trouver la matrice d'homographie entre deux ensembles de points correspondants.
    
    Arguments :
    - Matches : Liste des correspondances de points entre les deux images
    - BaseImage_kp : Points clés de l'image de référence
    - SecImage_kp : Points clés de l'image secondaire
    
    Retourne :
    - HomographyMatrix : Matrice d'homographie calculée
    - Status : Statut de la correspondance
    '''
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)
    BaseImage_pts = np.array(BaseImage_pts, dtype=np.float32)
    SecImage_pts = np.array(SecImage_pts, dtype=np.float32)
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)
    np.savetxt(str(os.path.dirname(os.path.dirname((__file__)))) + "/bloc/matrice_homographique.txt", HomographyMatrix,
               fmt='%f')
    return HomographyMatrix, Status


def getNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    '''
    Fonction pour calculer la taille de la nouvelle image résultante après transformation homographique 
    et mettre à jour la matrice d'homographie en conséquence.
    
    Arguments :
    - HomographyMatrix : Matrice d'homographie
    - Sec_ImageShape : Forme de l'image secondaire
    - Base_ImageShape : Forme de l'image de référence
    
    Retourne :
    - Nouvelle taille de l'image
    - Correction à appliquer à la matrice d'homographie
    - Matrice d'homographie mise à jour
    '''
    (Height, Width, _) = Sec_ImageShape
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)
    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))
    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)

    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.array([[0, 0], [Width - 1, 0], [Width - 1, Height - 1], [0, Height - 1]], dtype=np.float32)
    NewFinalPoints = np.float32(np.array([x, y]).transpose())
    HomographyMatrix = cv2.convertMaps(OldInitialPoints, np.array(NewFinalPoints, dtype=np.float32), cv2.CV_32F)
    return [New_Height, New_Width], Correction, HomographyMatrix