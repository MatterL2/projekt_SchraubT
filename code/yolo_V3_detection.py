import cv2
from ultralytics import YOLO

# 1. Charger votre modèle de segmentation entraîné
model = YOLO("runs/segment/train/weights/best.pt")

# 2. Ouvrir la webcam (0 représente généralement la webcam principale ou la caméra USB)
# Si vous utilisez une caméra de Raspberry Pi ou un autre flux, vous pouvez remplacer 0 par le chemin ou l'index de la caméra
cap = cv2.VideoCapture(1)

if not cap.isOpened():
  print("Erreur : Impossible d'ouvrir la caméra.")
  exit()

print(
    "Appuyez sur la touche 'q' de votre clavier pour quitter la vidéo en direct."
)

while cap.isOpened():
  # Lire une image (frame) de la webcam
  success, frame = cap.read()

  if not success:
    print(
        "Erreur : Impossible de lire le flux vidéo (fin de flux ou caméra"
        " débranchée)."
    )
    break

  # 3. Lancer l'inférence (IA) directement sur l'image de la webcam
  # stream=True est recommandé pour le flux vidéo en continu (optimisation de la mémoire)
  results = model(frame, stream=True)

  # 4. Traiter les résultats et dessiner les prédictions sur l'image
  for result in results:
    # Ultralytics fournit une méthode intégrée très pratique 'plot()'
    # qui dessine automatiquement les masques, les boîtes et les classes reconnues sur l'image.
    annotated_frame = result.plot()

    # 5. Afficher l'image annotée dans une fenêtre
    cv2.imshow("Detection de vis en direct - YOLOv8", annotated_frame)

  # Quitter la boucle si l'utilisateur appuie sur la touche 'q'
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Libérer la caméra et fermer toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()