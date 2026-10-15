import cv2
import numpy as np
import time 

dernier_affichage = time.time()

cap = cv2.VideoCapture(1)

while True:
  temps_actuel = time.time()

  ret, frame = cap.read()
  if not ret:
    break

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  lower_yellow = np.array([10, 40, 78])
  upper_yellow = np.array([54, 255, 255])

  # Création du masque pour la vis à bois
  mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

  # Recherche des contours sur le masque jaune
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  vis_a_bois_trouvee = False

  for c in contours:
    # On définit une surface minimale pour éviter de prendre en compte les petits parasites (bruit)
    area = cv2.contourArea(c)
    if area > 500:  # Ajuste cette valeur si besoin selon la distance de ta caméra
      vis_a_bois_trouvee = True
      # Optionnel : dessiner un cadre vert autour de la vis à bois sur l'image
      x, y, w, h = cv2.boundingRect(c)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # Affichage dans la console
  if temps_actuel - dernier_affichage >= 1.0:
    dernier_affichage = temps_actuel
    if vis_a_bois_trouvee:
      print("Vis à bois détectée !")
      print(area)
    else:
      print("Vis normale (ou aucun objet)")

  result = cv2.bitwise_and(frame, frame, mask=mask)
  cv2.imshow("frame", result)
  cv2.imshow("video avec cadre", frame)  # Pour voir le cadre vert

  if cv2.waitKey(1) == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()