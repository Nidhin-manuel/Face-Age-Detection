# Face Age Estimation Using Transfer Learning and Semi-Supervised Regression

This project focuses on improving face age prediction accuracy by combining transfer learning and semi-supervised learning techniques. It uses the UTKFace dataset, a large-scale real-world face dataset annotated with age, gender, and ethnicity information.

---

## 📚 Project Overview

- **Objective:**  
  To explore whether a **transfer learning-based semi-supervised regression model** can improve face age detection accuracy compared to traditional supervised learning methods.

- **Research Question:**  
  Can leveraging **pseudo-labeling techniques** and **transfer learning models (ResNet-50)** enhance the generalization of age estimation tasks?

- **Dataset:**  
  UTKFace Dataset ([Official Link](https://susanqq.github.io/UTKFace/))  
  - ~24,000 labeled facial images
  - Age range: 0–116 years
  - Annotations: age, gender, ethnicity

---

## ⚙️ Technologies and Tools

- Python 3.11
- TensorFlow/Keras
- Matplotlib, NumPy, Pandas
- Google Colab (GPU-backed runtime)
- GitHub for version control

---

## 🛠️ Project Workflow

1. **Data Preprocessing**
   - Filter images using UTKFace naming pattern
   - Resize images to (64x64)
   - Normalize pixel values
   - Data augmentation (flip, rotation, translation)

2. **Baseline Model**
   - Frozen ResNet-50 as feature extractor
   - Regression head to predict age
   - Trained using supervised learning (Baseline MAE: ~14.87 years)

3. **Fine-Tuning**
   - Unfroze top ResNet layers
   - Lowered learning rate (1e-5)
   - Improved model generalization (Fine-tuned MAE: ~13.71 years)

4. **Semi-Supervised Learning**
   - Pseudo-labeled 500 additional samples
   - Retrained model with mixed real + pseudo-labeled data
   - Semi-supervised MAE: ~14.15 years

5. **Evaluation**
   - MAE (Mean Absolute Error) and RMSE used
   - Comparison across Baseline, Fine-tuned, and Semi-supervised stages
   - Visualizations: MAE curves, prediction samples, error plots

---

## 📈 Results

| Model Phase        | Validation MAE (Years) |
|--------------------|-------------------------|
| Baseline (Frozen ResNet-50) | 14.87 |
| Fine-Tuned (Top layers unfrozen) | 13.71 |
| Semi-Supervised (with pseudo-labeled data) | 14.15 |

---

## 📊 Key Visualizations

- Model Training Histories (MAE vs Epochs)
- Age Distribution in UTKFace Dataset
- Actual vs Predicted Age Samples
- Error Analysis by Age Group
- Performance Comparison Table

---

## 🏆 Contributions

- Explored the effectiveness of semi-supervised learning in face age estimation.
- Reduced reliance on fully labeled datasets.
- Provided visual and quantitative comparisons between learning stages.

---

## 📎 References

- UTKFace Dataset by Susan Liang ([Link](https://susanqq.github.io/UTKFace/))
- ResNet-50 Architecture ([He et al., 2016](https://arxiv.org/abs/1512.03385))
- Semi-Supervised Learning Concepts ([Smailagic et al., 2023](https://arxiv.org/abs/2301.09589))
- Transfer Learning Principles ([MathWorks, 2023](https://www.mathworks.com/solutions/deep-learning/transfer-learning.html))

*(Full list of references available in the Final Project Report)*

---

## 📬 Contact

For any queries, contact:  
**Nidhin Manuel**  
**University of Hertfordshire**  
**Email: nidhinmanuelcx@gmail.com **

---
