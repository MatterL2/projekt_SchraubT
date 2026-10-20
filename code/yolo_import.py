from roboflow import Roboflow
rf = Roboflow(api_key="RlstJctkaBmNZta3lbfL")
project = rf.workspace("louiss-workspace-woekp").project("screw_v2")
version = project.version(1)
dataset = version.download("yolov8")