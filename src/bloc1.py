import cv2
import os

def redimension_et_blur(adresse_image,nb_image,gaussian_blur=True):
    '''
    Cette fonction reçoit en argument l'adresse d'une image, un entier et un booléen.
    Si cette image contient 1 milion de pixels ou plus elle est redimensionnée, réduite, tout en prenant en compte le rapport de proportionalité
    entre la longueur et la largeur de cette image : Si elle plus large que longue, etc ...
    Le booléen représente si on doit appliquer ou non le flou gaussien.
    La fonction enregistre enfin l'image dans le répertoire images de l'algorithme d'assemblage d'images.
    L'argument de type entier de la fonction permet d'enregistrer l'image avec le bon numéro (pour l'ordre d'assemblage).
    '''
    image = cv2.imread(adresse_image)
    h, w = image.shape[0], image.shape[1]
    image_tmp = image
    if h*w>=1e6:
        if h>w:
            image = cv2.resize(image,( 600,int(h/w)*720))
        elif h<w:
            image = cv2.resize(image,(720*int(w/h), 600))
        else:
            image = cv2.resize(image,(720,720))
    if gaussian_blur:
        image = cv2.medianBlur(image,3)
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/images/image"+str(nb_image)+".png",image)

def p_cles_et_descripteurs(image):
    '''
    Cette fonction prend en argument une image.
    La fonction uilise la fonctionSIFT d'open cv2 pour renvoyer les points clés et descripteurs de cette dernière. 
    '''
    sift = cv2.SIFT_create()
    pcles, descripteurs = sift.detectAndCompute(image, None)
    return pcles, descripteurs