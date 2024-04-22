import cv2
from bloc2 import *
from bloc3 import *

def images_avec_correspondances(matches,image1,image2,pcles1,pcles2):
    ''' 
    Cette fonction prend en paramètre les correspondances, 'matches' et les points clés, pcles1 et pcles2, entre image1 et image2. 
    D'abord la fonction met les deux images l'un à coté de l'autre , puis cherche les points commun et relie ces derniers. 
    Cela permet de vérifier visuellement les points communs des deux images.
    La fonction retourne une image sur lequel on voie les deux images, avec leurs points communs et des droites qui relient ces deniers.
    '''
    height = min(image1.shape[0], image2.shape[0])

    image1_resized = cv2.resize(image1, (image1.shape[1], height))
    image2_resized = cv2.resize(image2, (image2.shape[1], height))

    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    result_image = cv2.hconcat([image1_resized, image2_resized])

    for match in matches:
        for m in match:
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = pcles1[img1_idx].pt
            (x2, y2) = pcles2[img2_idx].pt
            x2 += image1.shape[1]
            cv2.line(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/bloc/img_matches.jpg", result_image)

def assemblage(BaseImage, SecImage, BaseImage_kp, SecImage_kp, descriptors1, descriptors2):
    ''' 
    La fonction assemblage prend en paramètre deux images, leur points clés et leur descripteurs.
    La fonction considère la première image comme étant l'image de base puis elle fait appel à la fonction Matches pour trouver les correspondances entre les 2 images.
    Ensuite la fonction calcule la matrice homographique. 
    Puis la fonction getNewFrameSizeAndMatrix estime une nouvele taille pour l'image finale et une nouvelle matrice homographique. 
    Enfin la fonction warpPerspective d'open cv2 applique la matrice homographique sur la deuxième image puis colle l'image de base à l'image obtenu avec la
    matrice homographique.
    La fonction renvoie une image composée des deux images précédentes bien positionnées. 
    '''
    Matches = FindMatches(descriptors1, descriptors2)
    images_avec_correspondances(Matches,BaseImage,SecImage,BaseImage_kp,SecImage_kp)
    HomographyMatrix, _ = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    [New_Height, New_Width], Correction, NewHomographyMatrix = getNewFrameSizeAndMatrix(HomographyMatrix, SecImage.shape, BaseImage.shape)
    WarpedImage = cv2.warpPerspective(SecImage, HomographyMatrix, (New_Width, New_Height))
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/bloc/warped_img.jpg", WarpedImage)
    h, w, _ = BaseImage.shape
    WarpedImage[: h, :w] = BaseImage
    return WarpedImage