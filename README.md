# Bert-Contract-Analysis
A Research on Classification of Contract Terms Using BERT-Based Sentence-wise Attention

### Introduction
- 리걸테크 산업에서 자연어처리를 활용해 계약서를 분석하는 새로운 방법론 제시
- 사전 언어 학습 모델인 BERT를 활용해 문서 단위의 계약서를 분석

### Model Architecture
- 사전학습된 BERT의 Sentence-Embedding을 활용해 Sentence Encoder를 사용
- 문서 내 분류할 문장인 Target sentence 이외에 문장을 고려하기 위한 Sentence-wise Attention Layer를 사용
- 문장 분류 task이지만 문서 내 있는 문장들을 한꺼번에 분류하기 위해 Output Layer를 사용
![image01](https://user-images.githubusercontent.com/37866322/102349036-0b53c780-3fe6-11eb-8ff0-001d3969bd88.png)

### Dataset
> Statistical

![image](https://user-images.githubusercontent.com/37866322/102349685-10654680-3fe7-11eb-867c-c9af74f7f053.png)
> Dataset Prepare

![image](https://user-images.githubusercontent.com/37866322/102350101-ba44d300-3fe7-11eb-8a03-c269a88604f3.png)


### Result
- 근로계약서 성능

![image](https://user-images.githubusercontent.com/37866322/102349879-61753a80-3fe7-11eb-8fb5-48b5284f15f3.png)
- 사채인수계약서 성능

![image](https://user-images.githubusercontent.com/37866322/102350264-fd9f4180-3fe7-11eb-84d5-74351d1fedee.png)

### Contract Analysis Example
- 근로계약서 분류 결과 예시

![image](https://user-images.githubusercontent.com/37866322/102350503-5e2e7e80-3fe8-11eb-880d-37786f35932b.png)

- 사채인수계약서 분류 결과 예시
![image](https://user-images.githubusercontent.com/37866322/102350558-70102180-3fe8-11eb-92e9-185b7ae97bed.png)

