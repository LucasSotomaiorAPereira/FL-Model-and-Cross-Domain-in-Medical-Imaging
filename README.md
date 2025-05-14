# FL-Model-and-Cross-Domain-in-Medical-Imaging
Kidney stones significantly impact healthcare systems, with diagnosis typically requiring time-consuming Computed Tomography (CT) scan consultations between medics and
radiologists, often delaying patient care. Achieving a quick and
accurate diagnosis is essential to ensure timely and effective
treatment, which has motivated the development of Deep
Neural Network (DNN)-based approaches for automated kidney
stone detection. However, building effective models remains
challenging, as it often requires access to large and diverse
datasets that are siloed across institutions, and sharing such
medical data is rarely feasible due to strict privacy regulations
and patient confidentiality concerns. This paper proposes a
privacy-preserving Federated Learning (FL) framework that
enables multiple medical institutions to collaboratively train
a DNN model without sharing sensitive patient data. Each
institution trains a local model on its private dataset, and a
centralized trusted server securely aggregates model parameters. We evaluate our approach using abdominal CT scan
image datasets from two distinct institutions. Experimental
results demonstrate that our proposed model achieves high
classification accuracy within the same training environment,
with an F1-score of up to 0.94. In addition, in cross-dataset
evaluations, our approach outperforms traditional centralized
baselines, showing significantly lower performance degradation
while preserving patient privacy.

![image](https://github.com/user-attachments/assets/831f7e44-77c6-405e-b154-46b5f4036c33)
