from main import programme_assemblage
import cv2
import os
from skimage.metrics import structural_similarity
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="`multichannel` is a deprecated argument name for `structural_similarity`.*")

chemin_repertoire = os.path.dirname((__file__))

def image_rotation(image,angle=-30):
    '''
    Cette fonction prend en argument et un angle, x.
    Avec l'angle et une fonction d'open cv2 on calcule une matrice de rotation.
    Enfin la fonction renvoie l'image de départ qui a subit une rotation de x degrés.
    '''
    hauteur, largeur = image.shape[0], image.shape[1]
    matrice_rotation = cv2.getRotationMatrix2D((largeur/2,hauteur/2),angle,1)
    return cv2.warpAffine(image,matrice_rotation,(largeur,hauteur))

def deux_images(valeur_superposition) : 
    '''
    La fonction va chercher l'image dans le dossier image_metriques.
    La fonction prend un seul argument la valeur de superposition.
    On calcule l'axe vertical central de l'image, une colonne centrale.
    La première image est découpée de la colonne de début à la colonne centrale, l'axe vertical central, plus la superposition.
    La seconde image est découpée de la colonne centrale moins la superposition à la fin de l'image.
    La fonction retourne ces deux images.
    '''
    valeur_superposition = valeur_superposition/2
    image_depart = cv2.imread(str(chemin_repertoire)+"/../image_metriques/image_depart.png")
    hauteur, largeur, _ = image_depart.shape
    superposition = int(largeur*valeur_superposition)
    axe_central = largeur//2
    image1 = image_depart[:, :axe_central+superposition]
    image2 = image_depart[:, axe_central-superposition:]
    return image1, image2


def psnr(image1,image2):
    '''
    Calcul du PSNR 
    Cette fontion prend  en paramètre deux images. Elle vérifie si les deux images ont la même taille. 
    Si oui on calcule le psnr, sinon on redimensionne la deuxième image puis on calcule le psnr.
    Enfin elle renvoie le psnr.
    '''
    if image1.shape != image2.shape:
        h,w=image1.shape
        image2 = cv2.resize(image2, (w,h))
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def ssim(image1,image2):
    '''
    Cette fonction prend deux images et renvoie le ssim entre ces deux images.
    '''
    return structural_similarity(image1,image2,multichannel=True)

'''def trois_images(valeur_superposition):
    """
    La fonction prend un seul argument, la valeur de superposition.
    Elle découpe une image en trois parties selon des axes situés à 1/3 et 2/3 de la largeur de l'image.
    Les segments résultants ont des zones de superposition autour de ces axes.
    """
    # Charger l'image de départ
    image_depart = cv2.imread(str(chemin_repertoire)+"/../image_metriques/image_depart.png")
    hauteur, largeur, _ = image_depart.shape
    
    # Calculer la valeur de superposition
    superposition = int(largeur * valeur_superposition)

    # Calculer les axes verticaux pour le découpage
    axe_1 = largeur // 3  # Premier axe (1/3 de la largeur)
    axe_2 = 2 * largeur // 3  # Deuxième axe (2/3 de la largeur)

    # Première image (de début jusqu'à l'axe_1 avec superposition)
    image1 = image_depart[:, :axe_1 + superposition]

    # Deuxième image (de axe_1 moins superposition jusqu'à axe_2 avec superposition)
    image2 = image_depart[:, axe_1 - superposition: axe_2 + superposition]

    # Troisième image (de axe_2 moins superposition jusqu'à la fin)
    image3 = image_depart[:, axe_2 - superposition:]

    return image1, image2, image3'''


def programme_metrique(valeur_superposition, angle, gaussian_blur=True):
    '''
    Cette fonction permet de mesurer le bruit que l'algorithme d'assemblage d'images ajoute à une image, image_depart.
    Elle a 3 arguments, un premier est la valeur de superposition que l'on choisit pour les deux images que l'on va obtenir  à partir d'image départ.
    Le troisième argument permet de chosir si on veut ou pas appliquer une flou gaussien dans l'algorithme d'assemblage.
    Pour cela on calcule d'abord les deux images à partir de image_depart.
    Ensuite on fait 3 tests.
    Pour le premier on enregistre les deux images dans le dossier images de l'algorithme d'assemblage, on applique ce dernier.
    On calcule le ssim et le psnr entre image_depart et le resultat de l'algorithme d'assemblage.
    Pour le deuxième test on fait pareil mais la seconde image a subit une rotation de x degrés dans le sens horaire. x étant le deuxième argument de la fonction.
    Pour le troisème test on fait pareil mais la seconde image a subit une rotation de x degrés dans le sens antihoraire.
    '''
    image1, image2 = deux_images(valeur_superposition)

    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/images/image1.png",image1)
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/images/image2.png",image2)

    if not gaussian_blur:
        programme_assemblage(False)
    else :
        programme_assemblage()
    image_depart = cv2.imread(str(chemin_repertoire)+"/../image_metriques/image_depart.png")
    image_resultat = cv2.imread(str(chemin_repertoire)+"/../resultat/resultat.png")
    image_depart = cv2.resize(image_depart, (image_resultat.shape[1], image_resultat.shape[0]))
    print("Sans modification de l'image droite")
    print("SSIM : "+str(ssim(image_depart,image_resultat)))
    print("PSNR : "+str(psnr(image_depart,image_resultat))+"\n")
    
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/images/image2.png",image_rotation(image2,-angle))
    if not gaussian_blur:
        programme_assemblage(False)
    else :
        programme_assemblage()
    image_depart = cv2.imread(str(chemin_repertoire)+"/../image_metriques/image_depart.png")
    image_resultat = cv2.imread(str(chemin_repertoire)+"/../resultat/resultat.png")
    image_depart = cv2.resize(image_depart, (image_resultat.shape[1], image_resultat.shape[0]))
    print("Modification 30 degrés sens horaire de l'image droite")
    print("SSIM : "+str(ssim(image_depart,image_resultat)))
    print("PSNR : "+str(psnr(image_depart,image_resultat))+"\n")
    
    cv2.imwrite(str(os.path.dirname(os.path.dirname((__file__)))) + "/images/image2.png",image_rotation(image2,angle))
    if not gaussian_blur:
        programme_assemblage(False)
    else :
        programme_assemblage()
    image_depart = cv2.imread(str(chemin_repertoire)+"/../image_metriques/image_depart.png")
    image_resultat = cv2.imread(str(chemin_repertoire)+"/../resultat/resultat.png")
    image_depart = cv2.resize(image_depart, (image_resultat.shape[1], image_resultat.shape[0]))
    print("Modification 30 degrés sens antihoraire de l'image droite")
    print("SSIM : "+str(ssim(image_depart,image_resultat)))
    print("PSNR : "+str(psnr(image_depart,image_resultat))+"\n")

programme_metrique(0.1,30)
