import carla
from scenic.simulators.carla.model import Car
import torch


from premise.carla.cnn_pytorch_main import CNN


import numpy as np


class ConesCar(Car):
    current_front_img = None
    model = None

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.device = "cuda"

        if ConesCar.model is None:
            ConesCar.model = CNN().to(self.device)
            model_path = "premise/examples/cones_model_new_499.pth"
            ConesCar.model.load_state_dict(torch.load(model_path))
            ConesCar.model.eval()

        self.model = ConesCar.model

    def camera_callback(self, image):
        if not self.carlaActor.is_alive:
            self.front_cam.stop()
            self.front_cam.destroy()
            return
        self.current_front_img_raw = image
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
        array = array[:, :, :3]  #  Take only RGB
        cropped_img = array[80:240, 20:300]  # [80:160,20:300]
        self.current_front_img = cropped_img[:, :, ::-1].copy()

    def startDynamicSimulation(self):
        cam_config = (
            self.carlaActor.get_world()
            .get_blueprint_library()
            .find("sensor.camera.rgb")
        )
        cam_config.set_attribute("image_size_x", str(320))
        cam_config.set_attribute("image_size_y", str(240))
        cam_config.set_attribute("fov", str(50))
        cam_location = carla.Location(2, 0, 1)
        cam_rotation = carla.Rotation(0, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)

        self.front_cam = self.carlaActor.get_world().spawn_actor(
            cam_config,
            cam_transform,
            attach_to=self.carlaActor,
            attachment_type=carla.AttachmentType.Rigid,
        )

        self.front_cam.listen(lambda image: self.camera_callback(image))

    def getDistanceCNN(self):

        img = torch.tensor(self.current_front_img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

        y_pred = self.model(img.to(self.device))

        value = y_pred.item()

        return value
