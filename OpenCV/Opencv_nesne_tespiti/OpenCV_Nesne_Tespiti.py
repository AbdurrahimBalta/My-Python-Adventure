#%% Kenar algılama

import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar

img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

med_val = np.median(img)
print(med_val)

low = int(max(0,(1-0.33)*med_val)) #sık kullanılan alt ve üst eşik belirme yöntemi
high = int(min(255,(1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

# blur işlemiyle kenarları azalttık tekrardan eşik hesaplıyoruz
blurred_img = cv2.blur(img, ksize = (5,5)) #kernelsize artırılarak kenarlar daha belirginleştirirelibilir
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")


med_val = np.median(blurred_img)
print(med_val)


low = int(max(0,(1-0.33)*med_val)) 
high = int(min(255,(1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")
#%% köşe algılama
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread("sudoku.jpg",0)
img = np.float32(img) #değişken tiğinde farklılıklar olmaması için ondalıklı sayılara çeviriyoruz 
print(img.shape)
plt.figure(),plt.imshow(img, cmap = "gray"), plt.axis("off")

# harris corner detection

dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)# blocksize = komşuluk boyutun #ksize = kutucuk boyutu #k = harris free parametr
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

dst = cv2.dilate(dst, None)
img[dst > 0.2 * dst.max()] = 1 # kutucuları genişletmek için bir hesap
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

# shi tomsai detection 

img = cv2.imread("sudoku.jpg",0)
img = np.float32(img) 
corners = cv2.goodFeaturesToTrack(img, 100, 0.001, 10) # 100 = max köşe sayısı,  0.001 = kalite seviyes , 10 iki köşe arasındaki mesafe 
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED)
#%% Kontur 
    
import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("contour.jpg",0)
plt.figure(), plt.imshow(img, cmap= "gray"),plt.axis("off")
img, contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) #iç ve dış ayıklamayı , köşelerin felan kodlanmasını sağlıyor

internal_countour = np.zeros(img.shape)   
external_countour  = np.zeros(img.shape)
    
for i in range(len(contours)):
    
    #external
    if hierarch[0][i][3] == -1:
        cv2.drawCountours(external_countour,contours, i, 255, -1)
    else: #internal
        cv2.drawCountours(internal_countour,contours, i, 255, -1)


plt.figure(), plt.imshow(external_countour, cmap= "gray"),plt.axis("off")
plt.figure(), plt.imshow(internal_countour, cmap= "gray"),plt.axis("off")

#%% renk ile nesne tespiti

import cv2
import numpy as np
from collections import deque

# nesne merkezini depolayacak veri tipi
buffer_size = 16
pts = deque(maxlen = buffer_size)

# mavi renk aralığı HSV
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

# capture
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success, imgOriginal = cap.read()
    
    if success: 
        
        # blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",hsv)
        
        # mavi için maske oluştur
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image",mask)
        # maskenin etrafında kalan gürültüleri sil
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Mask + erozyon ve genisleme",mask)
        
        # farklı sürüm için
        # (_, contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # kontur
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            
            # en buyuk konturu al
            c = max(contours, key = cv2.contourArea)
            
            # dikdörtgene çevir 
            rect = cv2.minAreaRect(c)
            
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            # kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # konturu çizdir: sarı
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)
            
            # merkere bir tane nokta çizelim: pembe
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
            
            # bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        # deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Orijinal Tespit",imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

#%% Şablon eşleme tepmlate matching
import cv2
import matplotlib.pyplot as plt

# template matching: sablon esleme

img = cv2.imread("cat.jpg", 0)
print(img.shape)
template = cv2.imread("cat_face.jpg", 0)
print(template.shape)
h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',# korelasyon medhodları
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    
    method = eval(meth)#eval stringi fonksiyona çevirir 'TM_CCOEFF_NORMED' ->> TM_CCOEFF_NORMED
    
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()
#%% özellik eşleme
import cv2
import matplotlib.pyplot as plt

#ana görüntü
chos = cv2.imread("chocolates.jpg",0)
plt.figure(),plt.imshow(chos, cmap = "gray"),plt.axis("off")
    
#aranacak görsel    
cho = cv2.imread("nestle.jpg",0)
plt.figure(),plt.imshow(cho, cmap = "gray"),plt.axis("off")

#orb tanımlayıcı
#köşe-kenar gibi nesneye ait featureler

orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos,None)


#bruf force matcher 
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#noktaları eşleştir
matches = bf.match(des1,des2)

#mesafeye göre sırala
matches = sorted(matches, key = lambda x: x.distance)

#eşleşen resimleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"), plt.title("orb")


#sift -> orbden daha iyi
sift = cv2.xfeatures2d.SIFT_create()

# bf
bf = cv2.BFMatcher()

kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)


matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []


for match1, match2 in matches:

    if match1.distance < 0.75 * match2.distance:
        guzel_eslesme.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, guzel_eslesme, None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")


#%% havza algoritması örnek 
import cv2
import matplotlib.pyplot as plt
import numpy as np


coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")


#lpf blurring 
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")


#grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap = "gray"), plt.axis("off")


# binary treshold
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")


# kontur
contours,hierarchy = cv2.findContours(coin_thresh.copy(),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours, i,(0,255,0),10)
        
plt.figure(), plt.imshow(coin), plt.axis("off")


#%%watershed (asıl havza algoritması)
import cv2
import matplotlib.pyplot as plt
import numpy as np

coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")


#lpf blurring 
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")


#grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap = "gray"), plt.axis("off")


# binary treshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# Açılma 

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off")

# nesneler arası distance bulma 
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")
 

# resmi küçült
ret, sure_foreground = cv2.threshold(dist_transform, 0.4 * np.max(dist_transform),255,0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")


# arka plan için resmi büyült
sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)

unknown = cv2.subtract(sure_background,sure_foreground)
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")

#bağlantı

ret, marker = cv2.connectedComponents(sure_foreground)

marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

#havza algoritması

marker = cv2.watershed(coin,marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")



# kontur
contours,hierarchy = cv2.findContours(marker.copy(),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours, i,(255,0,0),10)
        
plt.figure(), plt.imshow(coin), plt.axis("off")
#%%
import cv2
import matplotlib.pyplot as plt 


#içe aktar
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein,cmap = "gray"),plt.axis("off")

# sınıflandırıcı (yüz)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for(x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y), (x+w, y+h),(255,255,255),10)

plt.figure(), plt.imshow(einstein,cmap = "gray"),plt.axis("off")



#Barça
barce= cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barce,cmap = "gray"),plt.axis("off")


face_rect = face_cascade.detectMultiScale(barce)


for(x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y), (x+w, y+h),(255,255,255),10)

plt.figure(), plt.imshow(barce,cmap = "gray"),plt.axis("off")



#video

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame,minNeighbors = 3)
        
        
        for(x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,255),10)

        cv2.imshow("face detect",frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break 

cap.release(
cv2.destroyAllWindows        
#%% özel benzer özellikleri tespiti
"""
1) veri seti:
    n,p
2) cascade programı indir
3) cascade
4)cascade kullarak tespit algoritması
"""
import os 
import cv2

#resim deposu klasörü

path = "images"

#resim boyutu

imgWidth = 180
imgHeight = 120

#video capture

cap = cv2.VideoCapture(0) #kamera boyutları ve renk ayarları
cap.set(3,640)
cap.set(4,480)
cap.set(10,180)

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path + str(countFolder))

saveDataFunc()



count = 0 
countSave = 0

while True:
    
    success, img = cap.read()
    
    if success:
        
        img = cv2.resize(img,(imgWidth,imgHeight))
        
        if count % 5 == 0:
            cv2.imwrite(path + str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave += 1
            print(countSave)
            
        count += 1
        
        cv2.imshow("Image",img)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

import cv2

path = "cascade.xml"
objectName = "Kalem Ucu"

#%% Yaya tespiti
import cv2
import os

files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
    
print(img_path_list)

# hog tanımlayıcısı
hog = cv2.HOGDescriptor()
# tanımlayıcıa SVM ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
         
    cv2.imshow("Yaya: ",image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue
    
    
        
        
    
    
    












































































































    














        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

































    
    
    












































































