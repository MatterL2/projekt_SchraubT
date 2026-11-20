import time
import serial


class MegaSerialClient:
  def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=1.0):
    self.port = port
    self.baudrate = baudrate
    self.timeout = timeout
    self.ser = None
    self.derniere_commande = None
    self.dernier_envoi = 0.0

  def connect(self):
    self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
    time.sleep(2.0)  # Laisse l'Arduino Mega redémarrer à l'ouverture du port

  def send_line(self, message: str):
    if self.ser and self.ser.is_open:
      clean = message.strip() + "\n"
      self.ser.write(clean.encode("utf-8"))

  def send_detection(
      self,
      class_name: str,
      diam_mm: float,
      len_mm: float,
      min_interval: float = 0.5,
  ):
    """Envoie un ordre simple à l'Arduino Mega sans saturer le port série."""
    commande = f"PIECE:{class_name}:D={diam_mm:.1f}:L={len_mm:.1f}"
    maintenant = time.time()

    # Évite de renvoyer exactement la même chose trop vite
    if commande != self.derniere_commande or (
        maintenant - self.dernier_envoi >= min_interval
    ):
      self.send_line(commande)
      self.derniere_commande = commande
      self.dernier_envoi = maintenant

  def send_reject(self, reason: str = "uncertain"):
    self.send_line(f"REJET:{reason}")

  def read_line(self) -> str | None:
    """Lit une ligne envoyée par l'Arduino si disponible (non bloquant)."""
    if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
      return self.ser.readline().decode("utf-8", errors="ignore").strip()
    return None

  def close(self):
    if self.ser and self.ser.is_open:
      self.ser.close()