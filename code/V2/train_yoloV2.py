from ultralytics import YOLO

# 1. Charger un modèle YOLOv8 pré-entraîné (le modèle 'nano' ou 'n' est le plus rapide pour le Raspberry Pi)
model = YOLO("yolov8n.pt")

# 2. Lancer l'entraînement
results = model.train(
    data="screw_V2-1/data.yaml",  # Remplacez par le chemin exact vers votre fichier data.yaml téléchargé
    epochs=50,                           # Nombre de tours d'apprentissage (50 est un bon début pour un petit dataset)
    imgsz=512,                           # La taille d'image qu'on a configurée (512x512)
    batch=16                             # Taille du lot (réduisez à 8 si votre PC manque de mémoire RAM/GPU)
)