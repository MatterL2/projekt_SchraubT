import time
import cv2
import numpy as np
from ultralytics import YOLO

dernier_affichage = time.time()

piece_diametre = 25.75
calibration = None
diametre_piece_mes = []

# Liste simple pour stocker toutes les mesures brutes
liste_mesures_vis = []
MAX_HISTORIQUE = 30  # Taille de la fenêtre d'analyse

model = YOLO("runs/segment/train/weights/best.pt")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
  print("Erreur : Impossible d'ouvrir la caméra.")
  exit()

print(
    "Appuyez sur la touche 'q' de votre clavier pour quitter la vidéo en direct."
)

while cap.isOpened():
  temps_actuel = time.time()
  success, frame = cap.read()
  if not success:
    break

  annotated_frame = frame.copy()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

  results = model(frame, stream=True, verbose=False)

  for result in results:
    if result.masks is not None:
      for mask in result.masks.xy:
        points = np.array(mask, dtype=np.float32)
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (dim1, dim2), angle = rect
        longeur_pixel = max(dim1, dim2)

        # ON ENLÈVE LE FILTRE : On prend directement chaque longueur détectée
        # et on l'ajoute à notre historique brut
        liste_mesures_vis.append(longeur_pixel)
        if len(liste_mesures_vis) > MAX_HISTORIQUE:
          liste_mesures_vis.pop(0)

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(annotated_frame, [box], 0, (255, 0, 0), 2)
        cv2.putText(
            annotated_frame,
            f"{longeur_pixel:.1f} px",
            (int(center_x), int(center_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

  # Calibrage de la pièce
  cercles = cv2.HoughCircles(
      gray_blur,
      cv2.HOUGH_GRADIENT,
      dp=1,
      minDist=100,
      param1=100,
      param2=30,
      minRadius=20,
  )
  if cercles is not None:
    cercles = np.uint16(np.around(cercles))
    i = cercles[0, 0]
    if calibration is None:
      diametre_piece_mes.append(i[2] * 2)
      if len(diametre_piece_mes) >= 30:
        calibration = float(np.median(diametre_piece_mes))
        print(f"Calibration fixée : {calibration:.2f} pixels")
    cv2.circle(annotated_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

  # --- VOTRE LOGIQUE DE VALIDATION ET DE SÉLECTION MAX ---
  longeur_vis_validee = None

  if len(liste_mesures_vis) >= 10:
    valeurs_stables = []
    SEUIL_COMPTEUR = (
        5  # La valeur doit apparaître au moins 5 fois dans l'historique
    )

    for val in liste_mesures_vis:
      # Compter combien de mesures sont dans l'intervalle [val - 5, val + 5]
      compte = sum(1 for v in liste_mesures_vis if abs(v - val) <= 5)

      if compte >= SEUIL_COMPTEUR:
        valeurs_stables.append(val)

    # Si on a trouvé des valeurs stables (ex: 108 et 155), on garde la plus haute
    if valeurs_stables:
      longeur_vis_validee = max(valeurs_stables)

  # Calcul final et affichage
  if calibration is not None and longeur_vis_validee is not None:
    vis_mm = (longeur_vis_validee * piece_diametre) / calibration

    if temps_actuel - dernier_affichage >= 1.0:
      dernier_affichage = temps_actuel
      print(f"Taille vis validée & max (px): {longeur_vis_validee:.1f}")
      print(f"Longueur vis (mm): {vis_mm:.2f} mm")

  cv2.imshow("Detection de vis en direct - YOLOv8", annotated_frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()