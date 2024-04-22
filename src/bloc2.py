import cv2

def FindMatches(BaseImage_descrpt, SecImage_descrpt):
    '''
    La fonction prend en argument les descripteurs de 2 images.
    La fonction crée une objet de type  <<brute force match>> pour trouver toutes les correspondances entre ces 2 images.
    On boucle ensuite sur ces correspondances pour trouver les meilleures :
    Si la distance de la meilleure correspondance m est inférieure à 0.75 fois la distance de la deuxième meilleure correspondance n c'est une bonne correspondance.
    Enfin la fonction retourner une liste contenant les meilleures correspondances.
    '''
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_descrpt, SecImage_descrpt, k=2)
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])
    return GoodMatches