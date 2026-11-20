import time
import cv2
import numpy as np
from ultralytics import YOLO


class ScrewDetector:
    """Détection YOLO + mesure dimensionnelle (diamètre tige, longueurs) + calibration pièce."""

    def __init__(
        self,
        model_path: str = "code/V4/best.pt",
        coin_diameter_mm: float = 24.25,  # 50 centimes (2 euros = 25.75)
        default_pixel_to_mm: float = 24.25 / 108.0,  # Valeur initiale avant calibration
        confidence_threshold: float = 0.6,
    ):
        self.model = YOLO(model_path)
        self.coin_diameter_mm = coin_diameter_mm
        self.pixel_to_mm = default_pixel_to_mm
        self.confidence_threshold = confidence_threshold

        self.calibration_px = None
        self.coin_diameters_px = []

    def update_calibration(self, gray_blur, target_samples: int = 30, annotated_frame=None):
        """Détecte la pièce de monnaie et met à jour pixel_to_mm une fois target_samples atteint."""
        cercles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=20,
            maxRadius=130,
        )

        if cercles is None:
            return self.calibration_px

        cercles = np.uint16(np.around(cercles))
        c = cercles[0, 0]

        if self.calibration_px is None:
            self.coin_diameters_px.append(float(c[2] * 2))
            if len(self.coin_diameters_px) >= target_samples:
                self.calibration_px = float(np.median(self.coin_diameters_px))
                if self.calibration_px > 0:
                    self.pixel_to_mm = self.coin_diameter_mm / self.calibration_px

        if annotated_frame is not None:
            cv2.circle(annotated_frame, (c[0], c[1]), c[2], (0, 255, 0), 2)

        return self.calibration_px

    def analyze_frame(self, frame, calibration_samples: int = 30):
        """
        Analyse une image (frame) déjà capturée et retourne :
        - best_detection : dict avec class_name, confidence, shank_diam_mm, total_len_mm, shank_len_mm, infos
                         ou None si aucune détection ne dépasse confidence_threshold
        - annotated_frame : image annotée
        """
        annotated_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        # Mise à jour éventuelle de la calibration par pièce de monnaie
        self.update_calibration(
            gray_blur,
            target_samples=calibration_samples,
            annotated_frame=annotated_frame,
        )

        results = self.model(frame, stream=False, verbose=False)

        infos_classes_frame = []
        best_detection = None

        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i, mask in enumerate(result.masks.xy):
                    box = result.boxes[i]
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]  # Récupération propre du type de la vis
                    confidence = float(box.conf[0])  # Confiance entre 0.0 et 1.0
                    infos_classes_frame.append(f"{class_name}: {confidence:.2f}")

                    points = np.array(mask, dtype=np.float32)

                    # 1. Création d'un masque binaire pour isoler la vis actuelle
                    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_img, [np.int32(points)], 255)

                    # 2. Calcul des moments d'inertie pour trouver le centre et l'orientation
                    moments = cv2.moments(mask_img)
                    if moments["m00"] <= 0:
                        continue

                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]

                    mu20 = moments["mu20"] / moments["m00"]
                    mu02 = moments["mu02"] / moments["m00"]
                    mu11 = moments["mu11"] / moments["m00"]

                    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)

                    # 3. Récupération des pixels du masque et projection vectorielle (sans rotation d'image)
                    y_indices, x_indices = np.where(mask_img > 0)
                    if len(x_indices) <= 10:
                        continue

                    dx = x_indices - cx
                    dy = y_indices - cy

                    # u = axe longitudinal (longueur), v = axe transverse (largeur)
                    u = dx * cos_a + dy * sin_a
                    v = -dx * sin_a + dy * cos_a

                    # 4. Extraction du profil de largeur par tranches le long de u (pas de 1 pixel)
                    u_min, u_max = np.min(u), np.max(u)
                    bins = np.arange(u_min, u_max, 1.0)

                    widths = []
                    for j in range(len(bins) - 1):
                        mask_bin = (u >= bins[j]) & (u < bins[j + 1])
                        if np.any(mask_bin):
                            v_in_bin = v[mask_bin]
                            slice_width = np.max(v_in_bin) - np.min(v_in_bin)
                            widths.append(slice_width)

                    if len(widths) <= 10:
                        continue

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
                        widths,
                        np.ones(kernel_size) / kernel_size,
                        mode="same",
                    )

                    # --- 6. SÉPARATION TÊTE / TIGE ULTRA-ROBUSTE ---
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
                        seuil_transition = (head_diam_est + shank_diam_est) / 2
                        fin_tete_idx = 0
                        consecutive_count = 0
                        required_consecutive = 4

                        for idx in range(
                            int(len(smoothed_widths) * 0.05),
                            len(smoothed_widths) // 2,
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
                    else:
                        shank_length_px = total_length_px

                    shank_section_raw = widths[fin_tete_idx:]
                    shank_diameter_px = (
                        np.percentile(shank_section_raw, 97)
                        if len(shank_section_raw) > 0
                        else 0
                    )

                    # 7. Conversion en millimètres si la calibration est active
                    if self.calibration_px is not None and self.calibration_px > 0:
                        shank_diam_mm = shank_diameter_px * self.pixel_to_mm
                        total_len_mm = total_length_px * self.pixel_to_mm
                        shank_len_mm = shank_length_px * self.pixel_to_mm
                    else:
                        shank_diam_mm = 0.0
                        total_len_mm = 0.0
                        shank_len_mm = 0.0

                    # Affichage visuel sur l'image annotée
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

                    if confidence >= self.confidence_threshold and (
                        best_detection is None or confidence > best_detection["confidence"]
                    ):
                        best_detection = {
                            "class_name": class_name,
                            "confidence": confidence,
                            "shank_diam_mm": shank_diam_mm,
                            "total_len_mm": total_len_mm,
                            "shank_len_mm": shank_len_mm,
                            "infos": infos_classes_frame,
                        }

        return best_detection, annotated_frame


# --- Test autonome si ce fichier est lancé directement ---
if __name__ == "__main__":
    detector = ScrewDetector(model_path="code/V4/best.pt")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        exit()

    dernier_affichage = time.time()
    print("Appuyez sur la touche 'q' de votre clavier pour quitter la vidéo en direct.")

    while cap.isOpened():
        temps_actuel = time.time()
        success, frame = cap.read()
        if not success:
            break

        detection, annotated = detector.analyze_frame(frame)

        if detector.calibration_px is not None and temps_actuel - dernier_affichage >= 1.0:
            dernier_affichage = temps_actuel
            if detection is not None:
                msg = (
                    f"Diametre (mm): {detection['shank_diam_mm']:.2f} | "
                    f"Longueur total(mm): {detection['total_len_mm']:.2f} | "
                )
                if detection["class_name"] == "standard_screw":
                    msg += f"Longueur sous tete(mm): {detection['shank_len_mm']:.2f} | "
                msg += f"Détéctions [Classe: Confiance] -> {detection['infos']}"
                print(msg)
            else:
                print("not sure")

        cv2.imshow("Detection de vis en direct - YOLOv8", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()