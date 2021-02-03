#%%
import cv2

#görselleştirme
img = cv2.imread("messi5.jpg", 0)

#görselleştir
cv2.imshow("ilkresim",img)

k = cv2.waitKey(0) &0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("mess_gray.png",img)
    cv2.destroyAllWindows()
    
#%% Video içe aktarma
import cv2 
import time

video_name = "MOT17-04-DPM.mp4"

#video içe aktar: capture, cap

cap = cv2.VideoCapture(video_name)

print("genişlik",cap.get(3))
print("yükseklik",cap.get(4))

if cap.isOpened() == False:
    print("hata")

while True:
    ret, frame = cap.read()

    if ret == True:
        time.sleep(0.01) # uyarı: kullanmazsak çok hızlı akar
    
        cv2.imshow("video",frame)
    else: break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release() #☺ stop capture
cv2.destroyAllWindows
#%% Kamera açma video kaydı
import cv2
# capture

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#VİDEO KAYDET 

writer = cv2.VideoWriter("video_kaydı.mp4",cv2.VideoWriter_fourcc(*"DIVX"),20,(width,height))

while True:
    ret, frame = cap.read()
    cv2.imshow("Video",frame)
    
    #save
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows
    
    
#%% yeniden boyutlandırma

import cv2

img = cv2.imread("lenna.png")
print("picture size:",img.shape)
cv2.imshow("original",img)


#resized
imgResized = cv2.resize(img, (100,100))
print("resized img shape:", imgResized.shape)
cv2.imshow("img resized", imgResized)

#kırp

imgCropped = img[:200,:300] # height width
cv2.imshow("kirpikresm",imgCropped)

#%% Şekiller ve metin
import cv2
import numpy as np

#resim olustur


img = np.zeros((512,512,3), np.uint8) # siyah bir resim
cv2.imshow("Siyah", img)

# çizgi
# (resim ,başlangıç noktası, bitis noktası, renk)
cv2.line(img,(100,100),(100,300), (0,255,0)) #BGR (0,0,255)
cv2.imshow("Cizgi",img)


#dikdörtgen
#resim başlangıç bitiş renk
cv2.rectangle(img, (0,0), (256,256), (255,0,0),cv2.FILLED)
cv2.imshow("diktörtgen",img)

# çember eğer fılled yazarsan daire olur

cv2.circle(img, (300,300), 45,(0,0,255))
cv2.imshow("çeber",img)

# metin
# (resim baslangıç noktası font kalınlığı renk)
cv2.putText(img, "resim", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
cv2.imshow("Metin",img)
#%% Görüntülü birleştirme
import cv2
import numpy as np
#img içe aktar
img2 = cv2.imread("lenna.png")
cv2.imshow("Orginal", img2)

#horizontal
how = np.hstack((img2, img2))
cv2.imshow("horizontal",how)


#vertical

ver = np.vstack((img2,img2))
cv2.imshow("Dikey",ver)
#%% Perspektif çarpıtma
import cv2
import numpy as np

#içe aktar resim

img = cv2.imread("kart.png")
cv2.imshow("Original",img)

width = 400
height = 500

pts1 = np.float32([[230,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)

imgOutpu= cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("nihai resim",imgOutpu)
#%% Blending
import cv2 
import matplotlib.pyplot as plt

#karıştırma

img1 = cv2.imread("img1.jpg")
img1= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

img1 = cv2.resize(img1,(600,600))
img2 = cv2.resize(img2,(600,600))

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#karıştırılmış resim = alpha*img1 + beta*img2

blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma=0)

plt.figure()
plt.imshow(blended)
#%% görüntü eşikleme
import cv2
import matplotlib.pyplot as plt

#resmi içe aktar

img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.show()


#  eşikleme

_, thresh_img = cv2.threshold(img, thresh = 60, maxval= 255, type= cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.axis("off")
plt.show()

#uyarlamalı eşik değeri


thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()

#%% bulanıklaştırma

import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#blurrrein(detayı azaltır gürültüyü engeller)


img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("original"),plt.show()

"""

ortalama bulanıklaştırma yöntemi

"""

dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(),plt.imshow(dst2), plt.axis("off"), plt.title("ortlama  blur")


"""

gaussian blur

"""

gb = cv2.GaussianBlur(img, ksize=(3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("gauss blur")


"""
Medyan blur

"""

mb = cv2.medianBlur(img, ksize = 3)
plt.figure(),plt.imshow(mb), plt.axis("off"),plt.title("medyan blue")



def gaussianNoise(image):
    row,col,ch = image.shape
    mean = 0
    var = 0.05
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, (row,col,ch))
    
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


#içe aktar normalize et
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("original"),plt.show()

gaussianNoisyImage = gaussianNoise(img)
plt.figure(),plt.imshow(gaussianNoisyImage),plt.axis("off"),plt.title("asassafsf"),plt.show()


#gaussian blue

gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize=(3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb2), plt.axis("off"), plt.title("with gauss blue")

def saltPepperNoise(image): #row collumn channel
    row, col, ch = image.shape
    s_vs_p = 0.5   #SOFT/PEPPER
    
    amount = 0.004    
    
    
    noisy = np.copy(image)

    #salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1,int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    
    #pepper siyah
    num_pepper = np.ceil(amount * image.size * 1 - s_vs_p)
    coords = [np.random.randint(0, i-1, int(num_pepper))for i in image.shape]
    noisy[coords] = 0
    
    return noisy




spImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("sp ımage")



#median blur

mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
plt.figure(),plt.imshow(mb2), plt.axis("off"),plt.title("medyan blue")



#%% morflolojik operasyonlar 4 tane 
import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktar

img = cv2.imread("datai_team.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("orijinal ımg")

#erezyon [beyazcıkları azaltır]
kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 2)
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("erezyon")

# genişleme dilation [artırır]
result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("dilation")

# white noise oluşturma
whiteNoise = np.random.randint(0,2,size = img.shape[:2])
whiteNoise = whiteNoise*255
plt.figure(),plt.imshow(whiteNoise, cmap = "gray"),plt.axis("off"),plt.title("whitenoise")

noise_img = whiteNoise + img 
plt.figure(), plt.imshow(noise_img, cmap = "gray"), plt.axis("off"), plt.title("img + w ")



# açılma
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Acilma")


#black noise

blackNoise = np.random.randint(0,2,size = img.shape[:2])
blackNoise = blackNoise*-255
plt.figure(),plt.imshow(blackNoise, cmap = "gray"),plt.axis("off"),plt.title("blacknoise")

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= 245] = 0
plt.figure(),plt.imshow(black_noise_img, cmap = "gray"),plt.axis("off"),plt.title("filterblacknoise")

#kapatma
closing =cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE,kernel)
plt.figure(),plt.imshow(closing, cmap = "gray"), plt.axis("off"),plt.title("kapama")


# gradient kenar tespiti
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(),plt.imshow(gradient, cmap = "gray"), plt.axis("off"),plt.title("gradient")


#%% GÖRÜNTÜdeki yoğunluk veya renkteki yönlü bir değişikliktir kenar algılamada kullanulır
import cv2 
import matplotlib.pyplot as plt

#resmi içe aktar

img = cv2.imread("sudoku.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"),plt.axis("off"),plt.title("oroginal img")

# x gradyan
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap = "gray"),plt.axis("off"),plt.title("sobel imgx")


# y gradyan
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap = "gray"),plt.axis("off"),plt.title("sobel imgy")

# laplacian gradyan 

laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray"),plt.axis("off"),plt.title("laplacian img")
#%% Görüntü histogramı belirli bir görüntü için histograma bakılabilir
"""
Her bir ton değeri için piksel sayısını içerir
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktar

img = cv2.imread("red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(),plt.imshow(img_vis)


print(img.shape)

img_hist = cv2.calcHist([img], channels = [0],mask = None, histSize = [256], ranges = [0,256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist) 

color = ("b","g","r" )
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i],mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)

golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis)

print(golden_gate.shape)

mask = np.zeros(golden_gate.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap = "gray")

mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap = "gray")

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask = mask)
plt.figure(), plt.imshow(masked_img_vis, cmap = "gray")

masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask = mask)
masked_img_hist = cv2.calcHist([golden_gate], channels = [0],mask = mask, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(masked_img_hist)

# histogram eşitleme
# karşıtlık artırma

img = cv2.imread("hist_equ.jpg",0)
plt.figure(),plt.imshow(img, cmap = "gray")


img_hist = cv2.calcHist([img], channels = [0],mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap = "gray")

eq_img_hist = cv2.calcHist([eq_hist], channels = [0],mask = None, histSize = [256], ranges = [0,256])
plt.figure(),plt.plot(eq_img_hist)















































    
    
    
    
    