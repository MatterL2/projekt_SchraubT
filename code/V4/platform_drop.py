import cv2
import numpy as np


class PlatformDropDetector:
    """
    Détecte la présence et la chute d'une ou plusieurs vis sur la plateforme
    sans photo de référence, en filtrant par luminosité et taille de contour.
    """

    def __init__(
        self,
        roi: tuple[int, int, int, int] | None = None,
        dark_thresh: int = 110,
        min_area_px: int = 350,
        max_area_px: int = 8000,
        confirm_frames: int = 2,
    ):
        self.roi = roi
        self.dark_thresh = dark_thresh
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.confirm_frames = confirm_frames

        self.stable_count = 0
        self._candidate_count = None
        self._candidate_streak = 0

    def _extract_roi(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        if self.roi is None:
            rx = int(w * 0.12)
            ry = int(h * 0.12)
            rw = int(w * 0.76)
            rh = int(h * 0.76)
        else:
            rx, ry, rw, rh = self.roi
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))

        crop = frame[ry:ry + rh, rx:rx + rw]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # ROI circulaire
        mask = np.zeros_like(gray)
        center = (rw // 2, rh // 2)
        radius = min(rw, rh) // 2

        cv2.circle(mask, center, radius, 255, -1)

        gray = cv2.bitwise_and(gray, gray, mask=mask)

        return gray, mask, (rx, ry, rw, rh)

    def update(self, frame: np.ndarray) -> dict:
        annotated = frame.copy()

        gray_roi, roi_mask, (rx, ry, rw, rh) = self._extract_roi(frame)

        center_frame = (rx + rw // 2, ry + rh // 2)
        radius = min(rw, rh) // 2

        cv2.circle(
            annotated,
            center_frame,
            radius,
            (255, 200, 0),
            2,
        )

        # Isole les objets sombres (vis métalliques) sur le fond clair
        _, thresh = cv2.threshold(
            gray_roi,
            self.dark_thresh,
            255,
            cv2.THRESH_BINARY_INV,
        )

        # Ignore tout ce qui est hors du cercle
        thresh = cv2.bitwise_and(thresh, roi_mask)

        kernel = np.ones((3, 3), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        valid_contours = [
            c
            for c in contours
            if self.min_area_px <= cv2.contourArea(c) <= self.max_area_px
        ]

        raw_count = len(valid_contours)

        total_area = sum(
            int(cv2.contourArea(c))
            for c in valid_contours
        )

        if raw_count == self._candidate_count:
            self._candidate_streak += 1
        else:
            self._candidate_count = raw_count
            self._candidate_streak = 1

        object_appeared = False
        vis_tombee = False
        nb_tombees = 0

        if (
            self._candidate_streak >= self.confirm_frames
            and raw_count != self.stable_count
        ):
            if raw_count < self.stable_count:
                vis_tombee = True
                nb_tombees = self.stable_count - raw_count

            elif raw_count > self.stable_count:
                object_appeared = True

            self.stable_count = raw_count

        status_text = (
            f"VIS SUR PLATEFORME: {self.stable_count} ({total_area} px)"
            if self.stable_count > 0
            else f"AUCUNE VIS DETECTEE ({total_area} px)"
        )

        color = (
            (0, 255, 0)
            if self.stable_count > 0
            else (0, 0, 255)
        )

        cv2.putText(
            annotated,
            status_text,
            (rx, max(20, ry - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        for idx, c in enumerate(valid_contours, start=1):

            x_c, y_c, w_c, h_c = cv2.boundingRect(c)

            cv2.rectangle(
                annotated,
                (rx + x_c, ry + y_c),
                (rx + x_c + w_c, ry + y_c + h_c),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"#{idx}",
                (rx + x_c, max(15, ry + y_c - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        return {
            "object_count": self.stable_count,
            "object_present": self.stable_count > 0,
            "object_appeared": object_appeared,
            "vis_tombee": vis_tombee,
            "nb_tombees": nb_tombees,
            "total_area_px": total_area,
            "annotated_frame": annotated,
        }


if __name__ == "__main__":

    CAM_INDEX = 1

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la caméra {CAM_INDEX}.")
        exit()

    drop_detector = PlatformDropDetector(
        roi=None,
        dark_thresh=110,
        min_area_px=350,
        max_area_px=8000,
        confirm_frames=2,
    )

    print("Détection active. Appuyez sur 'q' pour quitter.")

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        state = drop_detector.update(frame)

        if state["vis_tombee"]:
            print(
                f"[EVENEMENT] {state['nb_tombees']} vis tombée(s) ! "
                f"Reste sur plateforme : {state['object_count']}"
            )

        cv2.imshow(
            "Camera Plateforme - Detection chute",
            state["annotated_frame"],
        )

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()