from mega_serial import MegaSerialClient


# Connexion série avec l'Arduino Mega
mega = MegaSerialClient(port="/dev/ttyACM0", baudrate=115200)
try:
  mega.connect()
  print("Connexion série avec l'Arduino Mega OK.")
except Exception as e:
  print(f"Attention : Arduino non connecté ({e}). Mode test caméra seul.")
  mega = None