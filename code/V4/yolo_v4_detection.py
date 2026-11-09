import time
import cv2
import numpy as np
from ultralytics import YOLO

dernier_affichage = time.time()

piece_diametre = 24.25 #50 centimes # 2 euros 25.75
pixelTomm = piece_diametre/326
calibration = None
diametre_piece_mes = []

model = YOLO("code/V4/best.pt")
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

  results = model(
      frame, stream=True, verbose=False
  )  # Prédiction YOLO (détection et segmentation)

  for result in results:
    if result.masks is not None and result.boxes is not None:
      for i, mask in enumerate(result.masks.xy):
        box = result.boxes[i]
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]  # Récupération propre du type de la vis

        
        points = np.array(mask, dtype=np.float32)

        # 1. Création d'un masque binaire pour isoler la vis actuelle
        mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [np.int32(points)], 255)



        # 2. Calcul des moments d'inertie pour trouver le centre et l'orientation
        moments = cv2.moments(mask_img)

        if moments["m00"] > 0:
          # Centre de gravité (centroid)
          cx = moments["m10"] / moments["m00"]
          cy = moments["m01"] / moments["m00"]

          # Moments centrés pour l'axe principal d'inertie
          mu20 = moments["mu20"] / moments["m00"]
          mu02 = moments["mu02"] / moments["m00"]
          mu11 = moments["mu11"] / moments["m00"]

          # Angle principal en radians
          angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
          cos_a = np.cos(angle_rad)
          sin_a = np.sin(angle_rad)

          # 3. Récupération des pixels du masque et projection vectorielle (sans rotation d'image)
          y_indices, x_indices = np.where(mask_img > 0)

          if len(x_indices) > 10:
            dx = x_indices - cx
            dy = y_indices - cy

            # u = axe longitudinal (longueur), v = axe transverse (largeur)
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            # 4. Extraction du profil de largeur par tranches le long de u (pas de 1 pixel)
            u_min, u_max = np.min(u), np.max(u)
            bins = np.arange(u_min, u_max, 1.0) #on fait un tabelau avec ecart de 1px a chque fois qui va de plus petit u au plus grand u pour qu apres on puisse avoir acces à chaque px 

            widths = []
            for i in range(len(bins) - 1):
              mask_bin = (u >= bins[i]) & (u < bins[i + 1])  #renvoie un tableau booleen comme ca on garde les index avec TRUE et on recuoere la valeur correspondant au TRUE
              if np.any(mask_bin):
                v_in_bin = v[mask_bin]
                slice_width = np.max(v_in_bin) - np.min(v_in_bin)  # on recupere les tranches de differentes largeur parce que dans le mask_bin on recupere chaque pixel qui a une certaine distance u du centre de gravite donc un tableau avec plusieurs valeurs c est pourquoio on prend max et min 
                widths.append(slice_width)

            if len(widths) > 10:
              widths = np.array(widths)


              # --- SÉCURITÉ ORIENTATION : S'assurer que la tête est toujours au début ---
              largeur_debut = np.mean(widths[:5])
              largeur_fin = np.mean(widths[-5:])
              if largeur_fin > largeur_debut:
                widths = widths[::-1]  # Inversion si la vis est à l'envers


              # 5. Lissage du profil pour éliminer l'effet des filets de vis
              kernel_size = max(3, len(widths) // 20)
              if kernel_size % 2 == 0:
                kernel_size += 1
              smoothed_widths = np.convolve(
                  widths, np.ones(kernel_size) / kernel_size, mode="same"   # calcul de la moyenne pour eviter d'avoir le filet, mode = same permet d avoir meme longeur tableau en sortie qu en entrée
              )

              # --- 6. SÉPARATION TÊTE / TIGE ULTRA-ROBUSTE ---
              # On prend le 90e percentile de la zone de tête (immunisé contre les pics de bruit/reflets)
              head_region = smoothed_widths[: int(len(smoothed_widths) * 0.25)]
              head_diam_est = (
                  np.percentile(head_region, 90) if len(head_region) > 0 else 0
              )

              shank_region = smoothed_widths[-int(len(smoothed_widths) * 0.3) :]
              shank_diam_est = (
                  np.median(shank_region) if len(shank_region) > 0 else 0
              )

              # Si la vis n'a pas de tête (goujon cylindrique)
              if head_diam_est < shank_diam_est * 1.15:
                fin_tete_idx = 0
              else:
                # Seuil de transition à mi-chemin entre la tête et la tige
                seuil_transition = (head_diam_est + shank_diam_est) / 2

                # Recherche du premier point sous le seuil qui se MAINTIENT (anti-bruit)
                fin_tete_idx = 0
                consecutive_count = 0
                required_consecutive = (
                    4  # Exige 4 pixels de suite sous le seuil
                )

                for idx in range(
                    int(len(smoothed_widths) * 0.05), len(smoothed_widths) // 2
                ):
                  if smoothed_widths[idx] < seuil_transition:
                    consecutive_count += 1
                    if consecutive_count >= required_consecutive:
                      fin_tete_idx = idx - required_consecutive + 1
                      break
                  else:
                    consecutive_count = 0

              # Calculs finaux en pixels
              total_length_px = u_max - u_min
              if class_name == "standard_screw":
                shank_length_px = total_length_px - fin_tete_idx
              else :
                shank_length_px = total_length_px


              shank_section_final = smoothed_widths[fin_tete_idx:]
              shank_diameter_px = (
                  np.median(shank_section_final)
                  if len(shank_section_final) > 0
                  else 0
              )


                # 7. Conversion en millimètres si la calibration est active
              if calibration is not None and calibration > 0:
                shank_diam_mm = (shank_diameter_px * piece_diametre) / calibration
                total_len_mm = (total_length_px * piece_diametre) / calibration
                shank_len_mm = (shank_length_px * piece_diametre) / calibration
              else:
                shank_diam_mm = 0
                total_len_mm = 0
                shank_len_mm = 0
                

              # Affichage visuel sur la frame
              cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
              cv2.putText(
                  annotated_frame,
                  f"Tige_diametre: {shank_diam_mm:.2f}mm",
                  (int(cx) - 40, int(cy) - 20),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (0, 255, 0),
                  2,
              )
              if class_name == "standard_screw":
                cv2.putText(
                    annotated_frame,
                    f"Long. sous tete: {shank_len_mm:.1f}mm",
                    (int(cx) - 50, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
              cv2.putText(
                  annotated_frame,
                  f"Total: {total_len_mm:.1f}mm",
                  (int(cx) - 50, int(cy)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (0, 255, 0),
                  2,
              )



  # Calibrage de la pièce (ex: pièce de monnaie)
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

  # Affichage périodique dans le terminal (toutes les secondes)
  if calibration is not None and temps_actuel - dernier_affichage >= 1.0:
    dernier_affichage = temps_actuel
    try:
      print(
          f"Diametre (mm): {shank_diam_mm:.2f} | Longueur total(mm): {total_len_mm:.2f} | ", end=""
            )
      if class_name =="standard_screw":
        print ( f"Longueur sous tete(mm): {shank_len_mm:.2f} | ", end="")
      print(f"Type: {class_name}")

    except NameError:
      pass

  cv2.imshow("Detection de vis en direct - YOLOv8", annotated_frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()