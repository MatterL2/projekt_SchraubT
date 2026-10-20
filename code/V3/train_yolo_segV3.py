from ultralytics import YOLO

# Charger le modèle de segmentation YOLOv8
model = YOLO("yolov8n-seg.pt")

print("Modèle chargé avec succès !")
# Lancer l'entraînement avec le batch et le GPU 1
results = model.train(
    data="screw_V3-1/data.yaml",
    epochs=50,
    imgsz=512,
    batch=16,  # Nombre d'images traitées en même temps par le GPU
    #device=1   # Votre RTX 40
)