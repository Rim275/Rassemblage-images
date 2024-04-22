import numpy as np
import cv2
import os

from bloc1 import *
from bloc4 import * 

import time

def programme_assemblage(gaussian_blur=True) : 
    '''
    C'est la fonction principale du programme d'assemblage d'images.
    D'abord la fonction va chercher toutes les images dans le répertoire images.
    Ensuite elle les redimensionne si nécessaire, et enregistre leur adresse dans le tableau des adresses des images.
    Ensuite la fonction boucle sur les images du tableau des images : 
    Elle fusionne les 2 premières, puis la 3ème avec la fusion de la 1ère et la 2ème, etc ...
    La fonction enregistre l'image résultat dans le dossier resultat.
    '''
    liste_images = []
    repertoire_projet = os.path.dirname(os.path.dirname((__file__)))
    fichiers = os.listdir(str(repertoire_projet+"/images"))

    nb_images = 0
    for element in fichiers:
        if element.endswith((".png",".jpg",".jpeg")):
            nb_images+=1
            adresse_image = os.path.join(str(repertoire_projet+"/images"),element)
            redimension_et_blur(adresse_image,nb_images)
            liste_images.append(adresse_image)

    image_gauche = cv2.imread(liste_images[0] )
    image_droite = cv2.imread(liste_images[1] )

    for i in range(1,nb_images):
        pcles_image_gauche, descripteurs_image_gauche = p_cles_et_descripteurs(image_gauche)
        pcles_image_droite, descripteurs_image_droite = p_cles_et_descripteurs(image_droite)
        image_fusion = assemblage(image_gauche, image_droite, pcles_image_gauche, pcles_image_droite, descripteurs_image_gauche, descripteurs_image_droite)
        if(i==nb_images-1):
            cv2.imwrite(str(repertoire_projet + "/resultat/") + "resultat.png", image_fusion)
            break
        else:
            chemin_resultat_tmp = str(repertoire_projet + "/images/") + "resultat_tmp.png"
            cv2.imwrite(chemin_resultat_tmp, image_fusion)
            image_gauche = cv2.imread(chemin_resultat_tmp)
            image_droite = cv2.imread(liste_images[i+1])

if __name__=="__main__":
    temps_debut = time.time()
    programme_assemblage()
    temps_fin = time.time()
    print("temps : "+str(temps_fin-temps_debut))