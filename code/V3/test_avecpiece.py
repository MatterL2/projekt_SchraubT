import cv2
from ultralytics import YOLO
import time 
import numpy as np

dernier_affichage = time.time()

piece_diametre = 25.75
calibration = 0 
diametre_piece_mes = []

# 1. Charger votre modèle de segmentation entraîné
model = YOLO("runs/segment/train/weights/best.pt")

# 2. Ouvrir la webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Appuyez sur la touche 'q' de votre clavier pour quitter la vidéo en direct.")

def pixel_in_mm(calibration,taille_vis_pix):
     return (taille_vis_pix * piece_diametre)/calibration 

while cap.isOpened():
    # Lire une image (frame) de la webcam
    success, frame = cap.read()

    # VÉRIFICATION IMMÉDIATE : Si la lecture a échoué, on quitte la boucle
    if not success:
        print("Erreur : Impossible de lire le flux vidéo (fin de flux ou caméra débranchée).")
        break

    temps_actuel = time.time()

    # 1. Transformer l'image de la webcam en niveaux de gris pour OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Appliquer un léger flou pour enlever le bruit (recommandé pour HoughCircles)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Par défaut, l'image à afficher est la copie de la frame brute de la webcam
    annotated_frame = frame.copy()

    # 3. Lancer l'inférence (IA) sur l'image
    results = model(frame, stream=True, verbose=False)

    # 4. Traiter les résultats YOLO (masques, etc.)
    

    # 5. Détecter la pièce de monnaie avec OpenCV (indépendant de YOLO)
    cercles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,param1=100,param2=30,minRadius=20)
    
    if cercles is not None:
        cercles = np.uint16(np.around(cercles))
        for i in cercles[0, :]:
            # Dessiner le cercle vert et le centre rouge sur l'image
            cv2.circle(annotated_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            diametre_piece_mes.append(i[1])
            calibration = max(diametre_piece_mes)
            vis_mm = pixel_in_mm(calibration,  )
            if temps_actuel - dernier_affichage >= 1.0:
                        dernier_affichage = temps_actuel
                        print(i[2])
            cv2.circle(annotated_frame, (i[0], i[1]), 10, (0, 0, 255), 3)

    # 6. AFFICHAGE UNIQUE : On affiche l'image finale dans la fenêtre (en dehors des boucles YOLO)
    cv2.imshow("Detection de vis en direct - YOLOv8", annotated_frame)

    # Quitter la boucle si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la caméra et fermer toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()