
import cv2
import numpy as np
import os
import glob

# -----------------------------
# PARAMETRES
# -----------------------------

CAMERA_INDEX = 1

# Nombre de coins INTERIEURS :
PATTERN_SIZE = (10, 7)

# Taille physique d'une case, en mm
SQUARE_SIZE_MM = 10.0

DOSSIER_IMAGES = "calibration_images"
FICHIER_CALIBRATION = "camera_calibration.npz"

os.makedirs(DOSSIER_IMAGES, exist_ok=True)

# Critères de raffinement des coins
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


def capturer_images():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la caméra.")
        return

    compteur = len(glob.glob(os.path.join(DOSSIER_IMAGES, "*.jpg")))

    print("\nCALIBRATION DE LA CAMERA")
    print("--------------------------------")
    print("Place la mire devant la caméra.")
    print("SPACE : enregistrer une image")
    print("Q : terminer la capture")
    print("--------------------------------\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur de lecture caméra.")
            break

        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            None
        )

        if found:
            cv2.drawChessboardCorners(
                display,
                PATTERN_SIZE,
                corners,
                found
            )

            cv2.putText(
                display,
                "MIRE DETECTEE",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                display,
                "Mire non detectee",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            f"Images : {compteur}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Capture calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if found:
                chemin = os.path.join(
                    DOSSIER_IMAGES,
                    f"image_{compteur:03d}.jpg"
                )

                cv2.imwrite(chemin, frame)
                compteur += 1

                print("Image enregistree :", chemin)
            else:
                print("Mire non detectee : image non enregistree.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def calculer_calibration():
    fichiers = glob.glob(
        os.path.join(DOSSIER_IMAGES, "*.jpg")
    )

    if len(fichiers) < 10:
        print(
            "\nERREUR : il faut au moins 10 images valides."
        )
        print("Je recommande plutot 15 a 25 images.")
        return

    # Coordonnees physiques des coins de la mire
    objp = np.zeros(
        (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    points_3d = []
    points_2d = []

    image_size = None

    for fichier in fichiers:
        image = cv2.imread(fichier)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            None
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                CRITERIA
            )

            points_3d.append(objp)
            points_2d.append(corners_refined)

            print("Mire detectee :", fichier)
        else:
            print("Mire non detectee :", fichier)

    if len(points_3d) < 10:
        print(
            "\nERREUR : moins de 10 images utilisables."
        )
        return

    # Calibration classique OpenCV
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            points_3d,
            points_2d,
            image_size,
            None,
            None
        )
    )

    np.savez(
        FICHIER_CALIBRATION,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        rms=rms
    )

    print("\n--------------------------------")
    print("CALIBRATION TERMINEE")
    print("--------------------------------")
    print("Images utilisees :", len(points_3d))
    print("Erreur RMS :", rms)
    print("Matrice camera :")
    print(camera_matrix)
    print("Coefficients de distorsion :")
    print(dist_coeffs)
    print("Fichier sauvegarde :", FICHIER_CALIBRATION)

    if rms > 1.0:
        print(
            "\nATTENTION : erreur RMS assez elevee. "
            "Refais des photos nettes et varie davantage "
            "les positions de la mire."
        )
    else:
        print(
            "\nErreur RMS encourageante. "
            "Il faut tout de meme verifier les images corrigees."
        )


if __name__ == "__main__":

    print("1 - Capturer des images de calibration")
    print("2 - Calculer la calibration avec les images existantes")

    choix = input("Ton choix : ")

    if choix == "1":
        capturer_images()

    elif choix == "2":
        calculer_calibration()

    else:
        print("Choix invalide.")
