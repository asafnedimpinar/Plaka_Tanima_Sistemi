import ast
import cv2
import numpy as np
import pandas as pd

# Çerçeve kenarlarına çizgi çizen fonksiyon
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Sol üst köşe
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Sol alt köşe
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Sağ üst köşe
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Sağ alt köşe
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# CSV dosyasından sonuçları oku
results = pd.read_csv('./test_interpolated.csv')

# Videoyu yükle
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

# Video kaydı için ayarlar
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)           # FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Genişlik
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Yükseklik
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# Her araç için en iyi plaka tespitini bulmak için sözlük
license_plate = {}

# Her bir araç için işlemler
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])

    # En yüksek skorlu plaka verisi
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_number_score'] == max_)]['license_number'].iloc[0]
    }

    # İlgili frame'e git
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    # Plakanın bbox'unu oku
    x1, y1, x2, y2 = ast.literal_eval(
        results[(results['car_id'] == car_id) &
                (results['license_number_score'] == max_)]['license_plate_bbox']
        .iloc[0]
        .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
    )

    # Plakayı kırp ve boyutlandır
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

# Frame okuma
frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Baştan başla

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1

    if ret:
        # O anki frame'e ait veriler
        df_ = results[results['frame_nmr'] == frame_nmr]

        for row_indx in range(len(df_)):
            # Araç bounding box'unu çiz
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox']
                .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Plaka bounding box'unu çiz
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox']
                .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Plaka kırpılmış görüntüsünü al
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
            H, W, _ = license_crop.shape

            try:
                # Plakayı aracın üstüne yapıştır
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                # Plakanın üstüne beyaz kutu çiz
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                # Yazı boyutunu hesapla
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                # Plaka numarasını yaz
                cv2.putText(
                    frame,
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    (0, 0, 0),
                    17
                )

            except:
                pass  # Hata olursa geç

        # Sonuç frame'ini kaydet
        out.write(frame)

        # (İstersen ekranda da gösterebilirsin)
        frame = cv2.resize(frame, (1280, 720))
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

# Video kayıtlarını kapat
out.release()
cap.release()
