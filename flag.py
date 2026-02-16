import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ================= AYARLAR =================
W_CAM, H_CAM = 640, 480
CLICK_DIST = 40        # Tıklama hassasiyeti
SMOOTHING = 0.2        
DRAW_SMOOTHING = 0.5   

# Renk Paleti (Neon - BGR)
COL_BG      = (30, 30, 30)
COL_CYAN    = (255, 200, 0)     
COL_MAGENTA = (200, 0, 200)     
COL_GREEN   = (0, 255, 0)       
COL_RED     = (50, 50, 255)     
COL_HOVER   = (255, 255, 255)   
COL_ERASER  = (0, 0, 0)         
COL_YELLOW  = (66, 175, 255) 

# ================= SİSTEM KURULUMLARI =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumeControl = cast(interface, POINTER(IAudioEndpointVolume))
try: current_bright = sbc.get_brightness()[0]
except: current_bright = 50
current_vol = 0.5

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Değişkenler
imgCanvas = np.zeros((H_CAM, W_CAM, 3), np.uint8)
xp, yp = 0, 0 
cx_curr, cy_curr = 0, 0
locked_hand_prev_coords = None 
is_fullscreen = True  # Başlangıç modu

# ================= GRAFİK FONKSİYONLARI =================
def draw_cool_cursor(img, x, y, is_clicking):
    color = COL_GREEN if is_clicking else COL_CYAN
    cv2.circle(img, (x, y), 5, color, cv2.FILLED)
    gap = 8 if is_clicking else 12
    len_ = 10
    cv2.line(img, (x, y-gap), (x, y-gap-len_), color, 2)
    cv2.line(img, (x, y+gap), (x, y+gap+len_), color, 2)
    cv2.line(img, (x-gap, y), (x-gap-len_, y), color, 2)
    cv2.line(img, (x+gap, y), (x+gap+len_, y), color, 2)

def draw_neon_rect(img, rect, color, filled=False, text=""):
    x, y, w, h = rect
    if filled:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (40, 40, 40), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    l = 10
    cv2.line(img, (x,y), (x+l, y), color, 2)
    cv2.line(img, (x,y), (x, y+l), color, 2)
    cv2.line(img, (x+w,y+h), (x+w-l, y+h), color, 2)
    cv2.line(img, (x+w,y+h), (x+w, y+h-l), color, 2)

    if text:
        scale = 1.0 if len(text) > 5 else 1.5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 2)
        tx, ty = x + (w-tw)//2, y + (h+th)//2 + 5
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_PLAIN, scale, (255,255,255), 2)

class Button:
    def __init__(self, rect, text, color):
        self.rect = rect
        self.text = text
        self.base_color = color
        self.hover = False

    def update(self, mx, my):
        x, y, w, h = self.rect
        self.hover = (x < mx < x+w and y < my < y+h)
        return self.hover

    def draw(self, img, force_active=False):
        color = COL_HOVER if (self.hover or force_active) else self.base_color
        is_filled = self.hover or force_active
        draw_neon_rect(img, self.rect, color, filled=is_filled, text=self.text)

# ================= BUTONLAR =================
# Ana Menü Butonları
btn_ses = Button((40, 100, 260, 70), "SES", COL_CYAN)
btn_par = Button((340, 100, 260, 70), "ISIK", COL_MAGENTA)
btn_med = Button((40, 200, 260, 70), "MUZIK", COL_GREEN)
btn_cnv = Button((340, 200, 260, 70), "CIZIM", COL_YELLOW)
btn_exit = Button((240, 380, 160, 50), "CIKIS", COL_RED)

# EKRAN BUTONU (SAĞ ALT KÖŞEYE TAŞINDI)
# Konum: x=510, y=420 (Ekranın sağ altı)
btn_scr  = Button((510, 420, 110, 40), "EKRAN", (150, 150, 150))

# Ortak
btn_back = Button((30, 30, 100, 40), "< GERI", (200,200,200))

# Medya
btn_prev = Button((100, 200, 100, 80), "|<<", COL_GREEN)
btn_play = Button((270, 200, 100, 80), "> ||", COL_GREEN)
btn_next = Button((440, 200, 100, 80), ">>|", COL_GREEN)

# Çizim (Çift El)
box_pen    = Button((20, 100, 80, 250), "KALEM", COL_YELLOW)
box_eraser = Button((540, 100, 80, 250), "SILGI", COL_RED)
btn_clear  = Button((500, 30, 100, 40), "SIL", COL_RED) # Sil butonunu eski yerine aldık (Ekran butonu aşağı indiği için)

app_mode = "MENU"
counter = 0

# ================= ANA DÖNGÜ =================
cap = cv2.VideoCapture(0)
cap.set(3, W_CAM)
cap.set(4, H_CAM)

# PENCERE AYARLARI
window_name = "Gemini Babaya Selam Olsun"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)

    # Çizim Katmanı
    if app_mode == "CANVAS":
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # --- AKILLI EL KİLİTLEME ---
    process_indices = []
    
    if results.multi_hand_landmarks:
        if app_mode == "CANVAS":
            process_indices = range(len(results.multi_hand_landmarks))
            locked_hand_prev_coords = None
        else:
            all_hands_centers = []
            for h in results.multi_hand_landmarks:
                cx, cy = int(h.landmark[9].x * W_CAM), int(h.landmark[9].y * H_CAM)
                all_hands_centers.append((cx, cy))

            if locked_hand_prev_coords is None:
                locked_hand_prev_coords = all_hands_centers[0]
                process_indices = [0]
            else:
                min_dist = float('inf')
                chosen_idx = -1
                for i, center in enumerate(all_hands_centers):
                    dist = math.hypot(center[0] - locked_hand_prev_coords[0], center[1] - locked_hand_prev_coords[1])
                    if dist < min_dist:
                        min_dist = dist
                        chosen_idx = i
                
                if chosen_idx != -1 and min_dist < 300:
                    locked_hand_prev_coords = all_hands_centers[chosen_idx]
                    process_indices = [chosen_idx]
                else:
                    locked_hand_prev_coords = None
    else:
        locked_hand_prev_coords = None

    # --- EL VERİLERİ ---
    hands_data = []
    if results.multi_hand_landmarks:
        for idx in process_indices:
            handLms = results.multi_hand_landmarks[idx]
            h, w, _ = img.shape
            cx, cy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h) 
            tx, ty = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h) 
            dist = math.hypot(tx - cx, ty - cy)
            hands_data.append({'x': cx, 'y': cy, 'click': dist < CLICK_DIST})

    mx, my, clicking = -100, -100, False
    if len(hands_data) > 0:
        mx, my = hands_data[0]['x'], hands_data[0]['y']
        clicking = hands_data[0]['click']

    # Debounce
    just_clicked = False
    if counter > 0:
        counter += 1
        if counter == 2: just_clicked = True
        if counter > 10: counter = 0
    if clicking and counter == 0: counter = 1

    # ================= 1. ANA MENÜ (BUTON BURADA) =================
    if app_mode == "MENU":
        draw_cool_cursor(img, mx, my, clicking)
        
        # Standart Menü Butonları
        for btn in [btn_ses, btn_par, btn_med, btn_cnv, btn_exit]:
            btn.update(mx, my)
            btn.draw(img)
        
        # EKRAN BUTONU - Sadece Ana Menüde
        btn_scr.update(mx, my)
        btn_scr.draw(img)
        
        cv2.putText(img, "KONTROL MERKEZI", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if just_clicked:
            if btn_ses.hover: app_mode = "VOLUME"
            elif btn_par.hover: app_mode = "BRIGHTNESS"
            elif btn_med.hover: app_mode = "MEDIA"
            elif btn_cnv.hover: 
                app_mode = "CANVAS"
                imgCanvas = np.zeros((H_CAM, W_CAM, 3), np.uint8)
            elif btn_exit.hover: break
            
            # EKRAN DEĞİŞTİRME MANTIĞI
            elif btn_scr.hover:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # ================= 2. SES =================
    elif app_mode == "VOLUME":
        draw_cool_cursor(img, mx, my, clicking)
        btn_back.update(mx, my)
        btn_back.draw(img)
        draw_neon_rect(img, (450, 100, 60, 300), COL_CYAN)
        fill = int(np.interp(current_vol, [0, 1], [0, 300]))
        cv2.rectangle(img, (450, 400 - fill), (510, 400), COL_CYAN, cv2.FILLED)
        cv2.putText(img, f"%{int(current_vol*100)}", (450, 90), cv2.FONT_HERSHEY_PLAIN, 2, COL_CYAN, 2)
        if just_clicked and btn_back.hover: app_mode = "MENU"
        if clicking and 400 < mx < 560 and 80 < my < 420:
            target = np.interp(my, [100, 400], [1, 0])
            current_vol = current_vol + (target - current_vol) * SMOOTHING
            volumeControl.SetMasterVolumeLevelScalar(current_vol, None)

    # ================= 3. IŞIK =================
    elif app_mode == "BRIGHTNESS":
        draw_cool_cursor(img, mx, my, clicking)
        btn_back.update(mx, my)
        btn_back.draw(img)
        draw_neon_rect(img, (450, 100, 60, 300), COL_MAGENTA)
        fill = int(np.interp(current_bright, [0, 100], [0, 300]))
        cv2.rectangle(img, (450, 400 - fill), (510, 400), COL_MAGENTA, cv2.FILLED)
        cv2.putText(img, f"%{int(current_bright)}", (450, 90), cv2.FONT_HERSHEY_PLAIN, 2, COL_MAGENTA, 2)
        if just_clicked and btn_back.hover: app_mode = "MENU"
        if clicking and 400 < mx < 560 and 80 < my < 420:
            target = np.interp(my, [100, 400], [100, 0])
            current_bright = current_bright + (target - current_bright) * SMOOTHING
            sbc.set_brightness(int(current_bright))

    # ================= 4. MEDYA =================
    elif app_mode == "MEDIA":
        draw_cool_cursor(img, mx, my, clicking)
        btn_back.update(mx, my)
        btn_back.draw(img)
        cv2.putText(img, "MEDYA PLAYER", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, COL_GREEN, 2)
        for btn in [btn_prev, btn_play, btn_next]:
            btn.update(mx, my)
            btn.draw(img)
        if just_clicked:
            if btn_back.hover: app_mode = "MENU"
            elif btn_prev.hover: pyautogui.press('prevtrack')
            elif btn_play.hover: pyautogui.press('playpause')
            elif btn_next.hover: pyautogui.press('nexttrack')

    # ================= 5. ÇİZİM =================
    elif app_mode == "CANVAS":
        btn_back.update(mx, my)
        btn_clear.update(mx, my)
        btn_back.draw(img)
        btn_clear.draw(img)

        if len(hands_data) > 0 and hands_data[0]['click']:
            if btn_back.hover: app_mode = "MENU"
            if btn_clear.hover: imgCanvas = np.zeros((H_CAM, W_CAM, 3), np.uint8)

        active_tool = None
        trigger_hand_idx = -1
        
        for i, hand in enumerate(hands_data):
            if box_pen.rect[0] < hand['x'] < box_pen.rect[0] + box_pen.rect[2] and \
               box_pen.rect[1] < hand['y'] < box_pen.rect[1] + box_pen.rect[3]:
                active_tool = "PEN"
                trigger_hand_idx = i
            elif box_eraser.rect[0] < hand['x'] < box_eraser.rect[0] + box_eraser.rect[2] and \
                 box_eraser.rect[1] < hand['y'] < box_eraser.rect[1] + box_eraser.rect[3]:
                active_tool = "ERASER"
                trigger_hand_idx = i

        if active_tool and len(hands_data) > 1:
            draw_idx = 1 if trigger_hand_idx == 0 else 0
            target_x = hands_data[draw_idx]['x']
            target_y = hands_data[draw_idx]['y']

            if cx_curr == 0 and cy_curr == 0:
                cx_curr, cy_curr = target_x, target_y
            else:
                cx_curr = cx_curr + (target_x - cx_curr) * DRAW_SMOOTHING
                cy_curr = cy_curr + (target_y - cy_curr) * DRAW_SMOOTHING
            
            draw_x, draw_y = int(cx_curr), int(cy_curr)

            color = COL_YELLOW if active_tool == "PEN" else COL_ERASER
            thickness = 8 if active_tool == "PEN" else 50
            
            if xp == 0 and yp == 0: xp, yp = draw_x, draw_y
            cv2.line(imgCanvas, (xp, yp), (draw_x, draw_y), color, thickness)
            xp, yp = draw_x, draw_y
            cursor_col = COL_YELLOW if active_tool == "PEN" else COL_RED
            cv2.circle(img, (draw_x, draw_y), 10, cursor_col, cv2.FILLED)
        else:
            xp, yp = 0, 0 
            cx_curr, cy_curr = 0, 0 
            for hand in hands_data:
                cv2.circle(img, (hand['x'], hand['y']), 8, (150, 150, 150), 2)

        box_pen.draw(img, force_active=(active_tool=="PEN"))
        box_eraser.draw(img, force_active=(active_tool=="ERASER"))
        info_text = "CIZIM AKTIF" if active_tool=="PEN" else ("SILGI AKTIF" if active_tool=="ERASER" else "ARAC SECIN")
        info_col = COL_YELLOW if active_tool=="PEN" else (COL_RED if active_tool=="ERASER" else (150,150,150))
        cv2.putText(img, info_text, (240, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, info_col, 2)

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hxCTF{b0sa_g3cmemis_1_0mur}
