# Real-Time Facial Expression Recognition for Enhanced Remote Therapy

**Authors:**  
Isabella F. Paolucci, Michael Amadi, Elisa de la Vega Ricardo, Eric S. Arnold  
School of Data Science, University of Virginia, Charlottesville, VA 22904  
Emails: ctx8bn@virginia.edu, mxg9xv@virginia.edu, vun4kt@virginia.edu, esa5ch@virginia.edu  

## Abstract

This paper presents a deep learning approach to enhancing remote therapy through real-time facial expression recognition. We develop and evaluate a suite of convolutional neural network (CNN) models—including custom architectures and transfer learning approaches—for classifying emotional states from facial images. Our system is implemented in PyTorch and trained on both FER-2013 and CK+ datasets, which are preprocessed, merged, and augmented to address class imbalance and variability.

We compare baseline and SMOTE-enhanced training strategies across multiple architectures, demonstrating that synthetic class balancing significantly improves F1 scores, particularly for underrepresented emotions. ResNet18 emerges as the top-performing model, achieving 59.00% accuracy and 53.23% macro-averaged F1 score on the combined dataset. Error analysis highlights persistent challenges in recognizing minority classes such as “contempt,” underscoring the need for targeted improvements in data representation and model generalization.

This work focuses on offline evaluation, providing a rigorous foundation for future integration into real-time telehealth systems. Our findings support the potential of CNN-based emotion recognition to improve therapist-client interactions, enhance remote care reliability, and contribute to emotionally intelligent digital health platforms.

**Keywords:** Facial Expression Recognition, Remote Therapy, Deep Learning, Telehealth, Emotion Detection

---

## 1. Motivation

**Enhancing Therapist-Client Interactions Through Real-Time Emotion Detection**

Effective therapy sessions rely on accurately perceiving a client’s emotional state. In remote sessions, subtle cues—such as micro-expressions—can be easily missed due to screen limitations, bandwidth constraints, and reduced visibility. Despite advances in video communication tools, therapists often report difficulty in gauging emotional depth and shifts during telehealth interactions.

An automated, real-time emotion recognition system offers the potential to continuously capture and analyze facial expressions, providing therapists with an objective emotional map of the session. Such systems could enhance clinical insight, reduce missed cues, and support more personalized and effective interventions. However, existing emotion recognition frameworks often lack the accuracy, interpretability, or real-time responsiveness required for sensitive therapeutic contexts.

This work aims to bridge that gap by investigating how state-of-the-art deep learning models can be adapted and evaluated for robust, real-time facial emotion detection in remote therapy settings.

---

## 2. Literature Review

### 2.1 Deep CNN Architectures and Benchmark Performance

Khaireddin and Chen (2021) achieved high accuracy on FER-2013 using VGGNet with optimized hyperparameters, showcasing CNNs' effectiveness in emotional expression recognition.

### 2.2 Dataset Generalization and Subtle Emotion Detection

MOL (2024) demonstrated generalizability and subtle emotional detection capabilities of CNNs using the CK+ dataset.

### 2.3 Evaluation Methodologies and Standardization

Paiva-Silva et al. (2016) identified inconsistencies in evaluation standards, emphasizing the need for standardized FER benchmarking methods.

### 2.4 Edge Computing for Real-Time Applications

Zhang et al. (2019) integrated CNNs with edge computing to improve performance in mobile and low-resource settings.

### 2.5 Transfer Learning and Cross-Dataset Validation

Ma (2024) validated CNNs across FER-2013 and CK+, confirming the value of transfer learning for emotion-specific adaptation.

### 2.6 Comprehensive CNN Evaluations and Best Practices

Giannopoulos et al. (2017) found CNNs outperformed traditional ML models, offering best practices in CNN training for FER.

### 2.7 Balancing Speed and Accuracy

Dar et al. (2022) proposed Efficient-SwishNet to balance computational efficiency and recognition accuracy for real-time use.

---

## 3. Methods

### 3.1 Datasets

We use FER-2013 (wild images) and CK+ (annotated muscle movement sequences) to train and validate emotion recognition models. The combined dataset helps maximize generalization and precision.

### 3.2 Methodology

#### Preprocessing and EDA

Face detection was done using OpenCV Haar cascades. Images were resized to 48×48, converted to grayscale, and normalized. SMOTE and weighted sampling were applied to mitigate class imbalance. Augmentation included flips, ±15° rotations, and ±10% scaling.

#### Model Architectures

- **MyCNN Series:** Custom CNNs increasing in depth (CNNv1 to CNNv6)
- **Transfer Learning:** ResNet18, VGG16, and DenseNet121 adapted for grayscale input and fine-tuned
- **Ma2024CNN:** Deep 5-layer CNN based on published architecture
- **Multi-Branch CNN:** Dual path (shape/texture) network fused for classification

#### Methodological Contributions

We assess the effectiveness of model designs under data imbalance constraints and explore baseline vs. enhanced training strategies.

---

## 4. Experiments

### 4.1 Training Protocol

All models used AdamW optimizer (lr = 1e-3, wd = 1e-4), batch size 32, and early stopping (patience = 3). Ma2024CNN used a batch size of 512 for 50 epochs.

### 4.2 Evaluation Metrics

Performance is measured using:
- Accuracy and macro-averaged F1 score
- Per-class precision, recall, and F1 score
- Confusion matrices

### 4.3 Results Summary

ResNet18 achieved the best results:
- Accuracy: 59.00%
- Macro F1: 53.23%

Custom models MyCNNv6 and Ma2024CNN also performed well with SMOTE. VGG16 showed overfitting and poor transferability. SMOTE consistently improved minority class performance.

---

## 5. Results

| Model         | Accuracy (Baseline) | F1 Score (Baseline) | Accuracy (SMOTE) | F1 Score (SMOTE) |
|---------------|---------------------|----------------------|------------------|------------------|
| MyCNNv1       | 42.49%              | 25.58%               | 44.57%           | 35.05%           |
| MyCNNv2       | 34.19%              | 16.42%               | 38.78%           | 26.73%           |
| MyCNNv3       | 42.88%              | 26.44%               | 47.46%           | 40.84%           |
| MyCNNv4       | 42.65%              | 25.54%               | 45.57%           | 39.05%           |
| MyCNNv5       | 45.56%              | 27.12%               | 44.49%           | 34.33%           |
| MyCNNv6       | 45.55%              | 30.53%               | 52.08%           | 46.41%           |
| Ma2024CNN     | 45.56%              | 31.33%               | 51.79%           | 46.23%           |
| DenseNet121   | 26.59%              | 15.34%               | 47.28%           | 38.90%           |
| VGG16         | 24.28%              | 4.88%                | 16.83%           | 3.60%            |
| ResNet18      | **51.41%**          | **40.09%**           | **59.00%**       | **53.23%**       |

---

## 6. Conclusion

### Discussion

This work demonstrates the potential of CNN-based models for emotion recognition in remote therapy. ResNet18 consistently performed best, with strong support for SMOTE-enhanced training. Custom models also showed strong improvements after augmentation and class rebalancing.

### Future Work

Key directions for future research include:
- Architectural enhancements to detect subtle emotions
- Emotion-specific data synthesis and transfer learning
- Deployment in clinical settings with real-time feedback loops

---

## Contributions

**Michael Amadi**  
Brainstorming and research (Milestone I), model development (Milestones I–III), abstract, motivation, literature review, final report compilation and presentation.

**Eric S. Arnold**  
Initial research, editing (Milestone II), Results and Conclusion writing (Milestone III), final presentation.

**Isabella Paolucci**  
Initial research and presentation (Milestone I), editing and submission (Milestone II), presentation slides, contributions, writing and editing for final report and presentation.

**Elisa de la Vega Ricardo**  
Initial research and presentation slides (Milestone I), model development (Milestones II–III), contributed to final report and presentation.

---

## References

References used in this report are included in the project bibliography file `references.bib` and cited throughout the text using standard citation practices.
