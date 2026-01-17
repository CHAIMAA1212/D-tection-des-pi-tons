import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
import onnxruntime as ort


class YOLO_Pred:

    def __init__(self, onnx_model, data_yaml):
        # Charger data.yaml
        with open(data_yaml, 'r') as f:
            data = yaml.load(f, Loader=SafeLoader)

        self.labels = data['names']
        self.nc = data['nc']

        # Charger le modéle ONNX
        self.session = ort.InferenceSession(
            onnx_model,
            providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.INPUT_WH_YOLO = 640
        self.CONF_TH = 0.4
        self.NMS_TH = 0.45

        # Couleur fixe pour "person"
        self.color = (0, 255, 0)

    def predictions(self, image):
        row, col, _ = image.shape

        # Image carrée
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Resize + normalisation
        img = cv2.resize(input_image, (self.INPUT_WH_YOLO, self.INPUT_WH_YOLO))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Inference
        preds = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0][0]

        boxes, confidences, class_ids = [], [], []

        x_factor = input_image.shape[1] / self.INPUT_WH_YOLO
        y_factor = input_image.shape[0] / self.INPUT_WH_YOLO

        for det in preds:
            obj_conf = det[4]
            if obj_conf < self.CONF_TH:
                continue

            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]

            score = obj_conf * class_conf

            if class_id == 0 and score > 0.25:
                cx, cy, w, h = det[:4]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(score))
                class_ids.append(class_id)

        # NMS
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.CONF_TH,
            self.NMS_TH
        )

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                conf = int(confidences[i] * 100)

                cv2.rectangle(
                    image, (x, y), (x + w, y + h),
                    self.color, 2
                )
                cv2.putText(
                    image,
                    f'person {conf}%',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.color,
                    2
                )

        return image
