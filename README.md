# FOSR-DA

**Open-Set Recognition of Communication Jamming Using Raw I/Q Data with Domain Adaptation**


FOSR-DA is a feature-enhanced deep learning framework for open-set recognition of communication jamming using raw I/Q data and adversarial domain adaptation.

---
## Requirements

- Python 3.9.12  
- PyTorch 1.12.1
  
## 📦 dataset
```
The datasets JPR2024 and JPR2016 are available in this repository, while JPR2018 can be obtained by contacting the authors.
```

## 🧪 Code Execution Flow

```
The order for running the code is as follows:

1、complexenhance.py：
Train the complex-valued autoencoder for I/Q feature enhancement.

2、complexmain.py：
Train the FOSR model for open-set jamming recognition.

3、domain adaptation.py
Adversarial domain adaptation training for FOSR model

4、opentest.py
Evaluate the trained model on test datasets and output performance metrics.
```

### License / 许可证

```
本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```
