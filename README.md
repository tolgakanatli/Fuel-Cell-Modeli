
# PEM Yakıt Hücresi Performans Modelleme ve Optimizasyon Kodu / PEM Fuel Cell Performance Modeling and Optimization Script

## 🇹🇷 Türkçe Açıklama

### 📌 Proje Adı
**PEM Yakıt Hücresi Performans Modelleme ve Optimizasyon Kodu**

### 🧪 Açıklama
Bu Python kodu, PEM (Proton Exchange Membrane) yakıt hücresi deneylerinden elde edilen maksimum güç yoğunluğu verilerini analiz etmek, farklı regresyon modelleri ile modellemek ve **Bayesian optimizasyon** yöntemi ile en uygun sıcaklık (x1) ve H₂:O₂ oranını (x2) belirlemek amacıyla geliştirilmiştir.

### 🔍 Kullanılan Modeller
- Lineer Regresyon  
- Polinom Regresyon (d=3)  
- Karar Ağacı Regresyon (derinlik 1–5)

En düşük hata değerine sahip model, tüm veri setine uygulanarak tahminler üretilir. Ardından, **Bayesian optimizasyon** ile maksimum güç yoğunluğu hedeflenerek optimum çalışma koşulları bulunur.

### ⚙️ Kullanılan Python Kütüphaneleri
- pandas  
- scikit-learn  
- bayes_opt

### 📊 Girdi Verisi
```
x1: Sıcaklık (°C)
x2: H2:O2 oranı (hacimsel)
y: Deneysel maksimum güç yoğunluğu (mW/cm²)
```

### 📈 Çıktılar
- Tahmin edilen vs. gerçek değerlerin karşılaştırması  
- En iyi modelin seçimi  
- Bayesian optimizasyon ile elde edilen en uygun koşullar  

### 🔒 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

## 🇬🇧 English Description

### 📌 Project Title
**PEM Fuel Cell Performance Modeling and Optimization Script**

### 🧪 Description
This Python script was developed to analyze experimental maximum power density data from a PEM (Proton Exchange Membrane) fuel cell, fit multiple regression models, and determine the optimal temperature (x1) and H₂:O₂ ratio (x2) using **Bayesian optimization**.

### 🔍 Models Used
- Linear Regression  
- Polynomial Regression (degree 3)  
- Decision Tree Regression (depth 1–5)

The best-performing model (with lowest error) is used to predict all values in the dataset. Then, **Bayesian optimization** is applied to find the optimal operating conditions that maximize power density.

### ⚙️ Python Libraries Used
- pandas  
- scikit-learn  
- bayes_opt

### 📊 Input Data
```
x1: Temperature (°C)
x2: H2:O2 ratio (by volume)
y: Experimental maximum power density (mW/cm²)
```

### 📈 Outputs
- Comparison of predicted vs. actual values  
- Best model selection  
- Optimal conditions via Bayesian optimization  

### 🔒 License
This project is licensed under the [MIT License](LICENSE).
