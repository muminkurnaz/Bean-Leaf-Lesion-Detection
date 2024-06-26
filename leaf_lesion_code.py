import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input/fasulye/fasulye_yapragi'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import cv2
import os
import matplotlib.pyplot as plt

# Veri setinin bulunduğu dizin
dataset_path = '/kaggle/input/fasulye/fasulye_yapragi'

# Klasör oluşturma - Gri görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/gri_resimler'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimleri griye dönüştürme ve görselleştirme
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        original_img = cv2.imread(img_path)
        
        gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        axes[1].imshow(gray_image, cmap='gray')
        axes[1].set_title('Gri Tonlamalı Resim')
        axes[1].axis('off')
        
        save_path = os.path.join(output_folder, f"gri_{filename}")
        cv2.imwrite(save_path, gray_image)
        
        plt.savefig(f"/kaggle/working/{filename}_output.png")
        
        plt.show()


# Klasör oluşturma - Yeniden boyutlandırılmış görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/boyutlandirilmis_resimler'
os.makedirs(output_folder, exist_ok=True)

# Yeni boyutlar
new_width = 350 
new_height = 350

# Tüm resimleri boyutlandırma işlemi
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Yeni boyutlara yeniden boyutlandırma işlemi
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Yeniden boyutlandırılmış resmi kaydetme
        save_path = os.path.join(output_folder, f"boyutlandirilmis_{filename}")
        cv2.imwrite(save_path, resized_img)
# Yeniden boyutlandırılmış resmi gösterme
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        ax.set_title('Yeniden Boyutlandırılmış Resim')
        ax.axis('off')
        plt.show()



output_folder = '/kaggle/working/esiklenmis_resimler'
os.makedirs(output_folder, exist_ok=True)

# Eşik değeri
threshold_value = 160

# Tüm resimlere eşikleme işlemi uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Griye dönüştürme
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Eşikleme işlemi
        _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Eşiklenmiş resmi gösterme
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal gri resim
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title('Orijinal Gri Resim')
        axes[0].axis('off')
        
        # Eşiklenmiş resim
        axes[1].imshow(thresholded_img, cmap='gray')
        axes[1].set_title('Eşiklenmiş Resim')
        axes[1].axis('off')
        
        # Eşiklenmiş resmi kaydetme
        save_path = os.path.join(output_folder, f"esiklenmis_{filename}")
        cv2.imwrite(save_path, thresholded_img)
        
        plt.show()



output_folder = '/kaggle/working/canny_kenar_tespit_resimler'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere kenar tespiti uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Kenar tespiti işlemi
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)  # Canny kenar tespiti
        
        # Kenar tespit edilmiş resmi gösterme
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Kenar tespit edilmiş resim
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title('Kenar Tespiti')
        axes[1].axis('off')
        
        # Kenar tespit edilmiş resmi kaydetme
        save_path = os.path.join(output_folder, f"kenar_{filename}")
        cv2.imwrite(save_path, edges)
        
        plt.show()


# Klasör oluşturma - Kenar tespit edilmiş görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/sobel_tespit_resimler'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere Sobel kenar tespiti uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Griye dönüştürme
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sobel kenar tespiti işlemi
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.sqrt(cv2.addWeighted(cv2.pow(sobel_x, 2.0), 1.0, cv2.pow(sobel_y, 2.0), 1.0, 0.0))
        
        # Sobel kenar tespit edilmiş resmi gösterme
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Sobel kenar tespit edilmiş resim
        axes[1].imshow(sobel_combined, cmap='gray')
        axes[1].set_title('Sobel Kenar Tespiti')
        axes[1].axis('off')
        
        # Sobel kenar tespit edilmiş resmi kaydetme
        save_path = os.path.join(output_folder, f"sobel_{filename}")
        cv2.imwrite(save_path, sobel_combined)
        
        plt.show()


# Klasör oluşturma - Kenar tespit edilmiş görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/laplace_tespit_resimler'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere Laplace kenar tespiti uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Griye dönüştürme
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplace kenar tespiti işlemi
        laplace = cv2.Laplacian(gray_img, cv2.CV_64F)
        
        # Laplace kenar tespit edilmiş resmi gösterme
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Laplace kenar tespit edilmiş resim
        axes[1].imshow(laplace, cmap='gray')
        axes[1].set_title('Laplace Kenar Tespiti')
        axes[1].axis('off')
        
        # Laplace kenar tespit edilmiş resmi kaydetme
        save_path = os.path.join(output_folder, f"laplace_{filename}")
        cv2.imwrite(save_path, laplace)
        
        plt.show()


output_folder_vertical = '/kaggle/working/dikey_ters_cevirme'
os.makedirs(output_folder_vertical, exist_ok=True)

# Tüm resimlere dikeyde ters çevirme işlemi uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Dikeyde ters çevirme işlemi
        flipped_img_vertical = cv2.flip(img, 0)
        
        # Görselleştirme (Orijinal ve Ters Çevrilmiş Resimler)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Ters çevrilmiş resim (dikey)
        axes[1].imshow(cv2.cvtColor(flipped_img_vertical, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Dikeyde Ters Çevrilmiş Resim')
        axes[1].axis('off')
        
        # Dikeyde ters çevrilmiş resmi kaydetme
        save_path_vertical = os.path.join(output_folder_vertical, f"flipped_vertical_{filename}")
        cv2.imwrite(save_path_vertical, flipped_img_vertical)
        
        plt.show()


# Klasör oluşturma - Histogram ve histogram eşitlenmiş görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/histogram_esitleme'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere histogram eşitleme işlemi uygulama ve histogramları gösterme
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Griye dönüştürme
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Histogram eşitleme işlemi
        img_equalized = cv2.equalizeHist(gray_img)
        
        # Görselleştirme (Orijinal, Histogram ve Histogram Eşitlenmiş Resimler)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Histogram
        axes[1].hist(gray_img.flatten(), bins=256, range=(0,256), color='blue', alpha=0.7)
        axes[1].set_title('Histogram')
        
        # Histogram eşitlenmiş resim
        axes[2].imshow(img_equalized, cmap='gray')
        axes[2].set_title('Histogram Eşitlenmiş Resim')
        axes[2].axis('off')
        
        # Histogram eşitlenmiş resmi kaydetme
        save_path = os.path.join(output_folder, f"equalized_{filename}")
        cv2.imwrite(save_path, img_equalized)
        
        # Histogram grafiğini kaydetme
        save_hist_path = os.path.join(output_folder, f"histogram_{filename}.png")
        plt.savefig(save_hist_path)
        
        plt.tight_layout()
        plt.show()


output_folder = '/kaggle/working/median_blur'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere Median blur uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Median blur işlemi
        median_blur_img = cv2.medianBlur(img, 5)  # Kernel boyutu: 5x5
        
        # Görselleştirme (Orijinal ve Median Blur Uygulanmış Resimler)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Median blur uygulanmış resim
        axes[1].imshow(cv2.cvtColor(median_blur_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Median Blur Uygulanmış Resim')
        axes[1].axis('off')
        
        # Median blur uygulanmış resmi kaydetme
        save_path = os.path.join(output_folder, f"median_blur_{filename}")
        cv2.imwrite(save_path, median_blur_img)
        
        plt.show()


# Klasör oluşturma - Gaussian blur uygulanmış görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/gaussian_blur'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere Gaussian blur uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Gaussian blur işlemi (Kernel boyutu: 5x5, Standart sapma: 0)
        gaussian_blur_img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Görselleştirme (Orijinal ve Gaussian Blur Uygulanmış Resimler)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Gaussian blur uygulanmış resim
        axes[1].imshow(cv2.cvtColor(gaussian_blur_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Gaussian Blur Uygulanmış Resim')
        axes[1].axis('off')
        
        # Gaussian blur uygulanmış resmi kaydetme
        save_path = os.path.join(output_folder, f"gaussian_blur_{filename}")
        cv2.imwrite(save_path, gaussian_blur_img)
        
        plt.show()


# Klasör oluşturma - Mean filtre uygulanmış görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/mean_filter'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere Mean filtre uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Mean filtre işlemi (Kernel boyutu: 5x5)
        mean_filter_img = cv2.blur(img, (7, 7))  # Kernel boyutu: 5x5
        
        # Görselleştirme (Orijinal ve Mean Filtre Uygulanmış Resimler)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Orijinal resim
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        
        # Mean filtre uygulanmış resim
        axes[1].imshow(cv2.cvtColor(mean_filter_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Mean Filtre Uygulanmış Resim')
        axes[1].axis('off')
        
        # Mean filtre uygulanmış resmi kaydetme
        save_path = os.path.join(output_folder, f"mean_filter_{filename}")
        cv2.imwrite(save_path, mean_filter_img)
        
        plt.show()


# Klasör oluşturma - Tuz ve biber gürültüsü uygulanmış görüntülerin kaydedileceği klasör
output_folder = '/kaggle/working/tuz_biber'
os.makedirs(output_folder, exist_ok=True)

# Tüm resimlere tuz ve biber gürültüsü uygulama
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        
        # Resmi yükleme
        img = cv2.imread(img_path)
        
        # Tuz ve biber gürültüsü ekleme
        salt_vs_pepper = 0.3  # Tuz ve biber oranı
        
        # Gürültü ekleme
        salt_pepper_img = np.copy(img)
        num_salt = np.ceil(salt_vs_pepper * img.size * 0.5)
        num_pepper = np.ceil(salt_vs_pepper * img.size * 0.5)
        
        # Tuz ekleme
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        salt_pepper_img[coords[0], coords[1], :] = 255
        
        # Biber ekleme
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        salt_pepper_img[coords[0], coords[1], :] = 0
        
        # Görselleştirme ve kaydetme (Orijinal ve Tuz/Biber Gürültülü Resimler)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Orijinal Resim')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(salt_pepper_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Tuz ve Biber Gürültülü Resim')
        axes[1].axis('off')
        
        # Tuz ve biber gürültülü resmi kaydetme
        save_path = os.path.join(output_folder, f"salt_pepper_{filename}")
        cv2.imwrite(save_path, salt_pepper_img)
        
        plt.show()
