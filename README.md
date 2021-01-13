# Kaggle

- https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview

# Ideas
- Lung segmentation 을 가지고 feature 뽑거나, 등등..?
    - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/notebooks?competitionId=10338&sortBy=voteCount
    - https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/data
    - https://www.kaggle.com/kmader/dsb-lung-segmentation-algorithm
        - 아래 External data Chest Xray dataset 활용?

# External data
- Chest Xray Masks and Labels
    - The dataset contains x-rays and corresponding masks. Some masks are missing so it is advised to cross-reference the images and masks.
- https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels
- https://www.kaggle.com/kmader/cyclegan-for-segmenting-xrays
- https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset
- https://www.kaggle.com/bonhart/chest-x-ray-eda-lung-segmentation

# Reference

- https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainall/seresnext50/train0.py

# 생각들
- 관이 제대로 삽입되어 있는 지 보려면, 몸속 장기들의 위치에 대한 정보도 같이 들어가줘야 판단하기 좋은거 아닐까
    - Lung mask 인풋으로..?