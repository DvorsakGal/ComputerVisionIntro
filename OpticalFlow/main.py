import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.registration as skreg


def analiziraj_opticni_pretok_2(video: np.ndarray) -> list[np.ndarray]:
    slike_poti = sorted(video)
    video_seq = []
    for pot in slike_poti:
        slika = plt.imread(pot)
        video_seq.append(slika)
    video = np.array(video_seq)

    # 1. Izračun povprečne slike ozadja
    slika_bg = video.mean(axis=0)  # Povprečje po časovni dimenziji

    # 2. Seznam za shranjevanje mask segmentacije
    segmentation_masks = []

    # 3. Parametri za segmentacijo
    threshold_value = 50  # Prag za segmentacijo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Jedro za morfološke operacije

    for n in range(video.shape[0]):
        # Trenutna slika in razlika od ozadja
        slika = video[n]
        slika_diff = np.abs(slika - slika_bg).mean(axis=2)  # Povprečje po barvnih kanalih

        # 4. Pragovna segmentacija
        _, segmented = cv2.threshold(slika_diff.astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)

        # 5. Morfološke operacije za izboljšanje maske
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)  # Zapiranje za odpravo lukenj
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)  # Odpiranje za odstranitev šuma

        # Shrani masko
        segmentation_masks.append(segmented)

    # Zbirka za konture na vsaki sliki
    filtered_contours_per_frame = []

    for n, mask in enumerate(segmentation_masks):
        # Najdi konture
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtriraj konture na podlagi površine
        min_area = 160  # Minimalna površina gibajočega elementa
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        filtered_contours_per_frame.append(filtered_contours)

    # Seznam za shranjevanje povezanih gibajočih elementov
    moving_elements = []

    # Prvi okvir
    prev_contours = filtered_contours_per_frame[0]

    # Določitev gibajočih elementov skozi sličice
    for i in range(1, len(filtered_contours_per_frame)):
        curr_contours = filtered_contours_per_frame[i]

        # Seznam za trenutno sličico
        curr_elements = []

        for curr_cnt in curr_contours:
            curr_rect = cv2.boundingRect(curr_cnt)
            matched = False

            for prev_cnt in prev_contours:
                prev_rect = cv2.boundingRect(prev_cnt)

                # Preveri, ali se pravokotnika prekrivata
                if not (curr_rect[0] + curr_rect[2] < prev_rect[0] or
                        prev_rect[0] + prev_rect[2] < curr_rect[0] or
                        curr_rect[1] + curr_rect[3] < prev_rect[1] or
                        prev_rect[1] + prev_rect[3] < curr_rect[1]):
                    matched = True
                    break

            # Dodaj trenutni element (lahko ga kasneje označiš kot nov, če ni povezave)
            curr_elements.append(curr_cnt)

        # Posodobi konture za naslednji korak
        prev_contours = curr_contours

        # Shrani trenutne elemente
        moving_elements.append(curr_elements)

    print("Gibajoči elementi so bili uspešno povezani!")

    # Željene analize:
    analize = []
    # Inicializacija optičnega pretoka za vsak par sličic
    flow_maps = []
    for n in range(video.shape[0] - 1):
        slika_0 = (video[n] * 255).mean(2).astype(np.uint8)
        slika_1 = (video[n + 1] * 255).mean(2).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(slika_0, slika_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_maps.append(flow)

    # Povprečen optični pretok za gibajoče elemente
    for i, (elements, flow) in enumerate(zip(moving_elements, flow_maps)):
        # Inicializiraj barvno sliko (HSV format)
        h, w = flow.shape[:2]
        hsv_img = np.zeros((h, w, 3), dtype=np.float32)
        hsv_img[..., 1] = 1  # Saturation na maksimum

        for cnt in elements:
            # Ustvari masko za trenutno konturo
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # Povprečni dx, dy znotraj konture
            dx_mean = np.mean(flow[..., 0][mask == 255])
            dy_mean = np.mean(flow[..., 1][mask == 255])

            # Pretvorba v amplitudo in smer
            amplitude = np.sqrt(dx_mean ** 2 + dy_mean ** 2)
            direction = np.arctan2(dy_mean, dx_mean)  # Smer v radianih

            # Pretvorba v barvno kodacijo (Hue in Value)
            hue = (direction * 180 / np.pi / 2) % 180  # Pretvori radiane v stopinje (Hue)
            value = np.clip(amplitude / amplitude.max(), 0, 1)  # Normalizacija amplitude

            # Barvna kodacija v območju konture
            hsv_img[..., 0][mask == 255] = hue  # Hue
            hsv_img[..., 2][mask == 255] = value  # Value

        # Pretvorba iz HSV v RGB za prikaz
        rgb_img = cv2.cvtColor((hsv_img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        analize.append(rgb_img)

    return analize

