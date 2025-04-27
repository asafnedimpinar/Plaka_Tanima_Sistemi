from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Sonuçları saklayacağımız sözlük
results = {}

# Sort algoritması ile nesne takibi için tracker nesnesi
mot_tracker = Sort()

# Modelleri yükle
coco_model = YOLO('yolov8n.pt')  # COCO veri seti üzerinde eğitilmiş genel nesne algılama modeli
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Plaka algılama modeli

# Video dosyasını yükle
cap = cv2.VideoCapture('./sample.mp4')

# Araç sınıf ID'leri (COCO veri setine göre): araba, motosiklet, otobüs, kamyon
vehicles = [2, 3, 5, 7]

# Frame okumaya başla
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Araçları tespit et
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Tespit edilen araçları takip et
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Plakaları tespit et
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Tespit edilen plakayı bir araca eşleştir
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Plaka bölgesini kırp
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Plaka görüntüsünü griye çevir ve eşik uygula
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Plaka numarasını oku
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    # Sonuçları kaydet
                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

# Tüm sonuçları CSV dosyasına yaz
write_csv(results, './test.csv')
