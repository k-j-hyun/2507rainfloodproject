# B2B 장마철 침수 예측 머신러닝 시스템

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> **예측 가능한 침수 사고 예방을 통한 사회/기업 전반의 공공이익 구조 구현**

## 프로젝트 개요

장마철 기상 데이터와 과거 침수 이력을 기반으로 지역별 침수 위험도를 예측하는 B2B 머신러닝 시스템입니다. 다중 모델 앙상블 기법을 활용하여 높은 예측 정확도를 달성하고, 실시간 API를 통해 비즈니스 파트너들에게 침수 예측 서비스를 제공합니다.

### 주요 목표
- **침수 사고 예방**: 사전 예측을 통한 피해 최소화
- **비즈니스 연계**: B2B 서비스로 다양한 산업 분야 지원
- **실시간 대응**: API 기반 실시간 예측 서비스 제공

---

## 주요 기능

### **다중 모델 앙상블 예측**
- RandomForest, XGBoost, LightGBM 기반 전통적 ML 모델
- CNN+LSTM 결합 딥러닝 모델
- Transformer 기반 시계열 예측 모델
- 모델별 성능 비교 및 최적 앙상블 구성

### **실시간 웹 서비스**
- Flask 기반 REST API 제공
- 직관적인 웹 인터페이스
- 실시간 데이터 시각화
- 지역별 위험도 맵 제공

### **종합 데이터 분석**
- 기상청 공공데이터 실시간 수집
- 과거 침수 이력 데이터 분석
- 뉴스 데이터 크롤링을 통한 검증 데이터 확보

---

## 기술 스택

### **Backend & ML**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)

### **Data Processing & Visualization**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

### **Database & Tools**
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)

---

## 데이터 소스

| 데이터 소스 | 내용 | 활용 방법 |
|------------|------|-----------|
| **공공데이터포털 ASOS** | 지상 시간별/일별 기상 데이터 | 강수량, 습도, 기압 등 기본 예측 변수 |
| **물정보포털** | 실제 침수 사례 데이터 | 타겟 변수 및 모델 검증 데이터 |
| **네이버 뉴스 API** | 침수 관련 뉴스 크롤링 | 데이터 검증 및 보조 지표 |
| **기상청 Open API** | 초단기예보, 동네예보, 기상특보 | 실시간 예측 서비스 |

---

## 설치 및 실행

### **요구사항**
```
Python >= 3.10
MySQL >= 8.0
```

### **빠른 시작**

1. **저장소 클론**
```bash
git clone https://github.com/k-j-hyun/2507rainfloodproject.git
cd 2507rainfloodproject
```

2. **가상환경 설정**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **데이터베이스 설정**
```bash
# MySQL 데이터베이스 생성
mysql -u root -p -e "CREATE DATABASE flood_prediction;"

# 환경변수 설정 (.env 파일 생성)
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=flood_prediction
```

5. **애플리케이션 실행**
```bash
python run.py
```

6. **웹 브라우저에서 확인**
```
- 모든 기능 구현 로컬 페이지
http://localhost:5000

- 데모페이지(하드코딩 웹페이지)
https://rainflood.onrender.com/
```

---

## 모델 성능

### **모델별 성능 비교**

| 모델 | 정확도 | F1-Score | 예측 시간 |
|------|--------|----------|-----------|
| **RandomForest** | 87.3% | 0.851 | 0.12s |
| **XGBoost** | 89.1% | 0.873 | 0.08s |
| **LightGBM** | 88.7% | 0.869 | 0.06s |
| **CNN+LSTM** | 91.2% | 0.896 | 1.24s |
| **Transformer** | 90.8% | 0.889 | 2.17s |
| **앙상블 모델** | **92.4%** | **0.908** | 0.73s |

### **최종 앙상블 구성**
- XGBoost (40%), CNN+LSTM (35%), Transformer (25%)
- Weighted Voting 방식 적용
- Cross-validation 5-fold 평균 성능 기준

---

## 프로젝트 시연

[![프로젝트 시연 영상](https://img.youtube.com/vi/xTN3h9XASdI/0.jpg)](https://youtu.be/xTN3h9XASdI)

**[시연 영상 보기](https://youtu.be/xTN3h9XASdI)**

---

## 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  ML Pipeline    │    │   Web Service   │
│                 │    │                 │    │                 │
│ • 공공데이터포털 │───▶│ • Data ETL      │───▶│ • Flask API     │
│ • 물정보포털     │    │ • Feature Eng.  │    │ • Web Interface │
│ • 뉴스 크롤링    │    │ • Model Train   │    │ • Visualization │
│ • 기상청 API    │    │ • Ensemble      │    │ • Real-time     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 프로젝트 구조

```
2507rainfloodproject/
├── data/                      # 데이터 파일
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── models/                # 학습된 모델
├── src/                       # 소스 코드
│   ├── data_collection/       # 데이터 수집 모듈
│   ├── preprocessing/         # 전처리 모듈
│   ├── models/               # ML 모델들
│   └── utils/                # 유틸리티 함수
├── web/                       # 웹 애플리케이션
│   ├── templates/            # HTML 템플릿
│   ├── static/               # CSS, JS, 이미지
│   └── app.py                # Flask 애플리케이션
├── notebooks/                 # Jupyter 노트북
│   ├── EDA.ipynb             # 탐색적 데이터 분석
│   ├── modeling.ipynb        # 모델링 과정
│   └── evaluation.ipynb      # 모델 평가
├── requirements.txt           # 의존성 목록
└── README.md                 # 프로젝트 문서
```

---

## 팀 구성

| 역할 | 담당자 | 주요 업무 |
|------|--------|-----------|
| **팀 리더 & ML Engineer** | 고정현 | 프로젝트 총괄, 모델 개발, API 구현 |
| **Data Engineer** | 팀원A | 데이터 수집 및 전처리 |
| **Frontend Developer** | 팀원B | 웹 인터페이스 개발 |
| **Data Analyst** | 팀원C | EDA 및 시각화 |

---

## 향후 개선 계획

- [ ] **실시간 스트리밍 파이프라인** 구축 (Apache Kafka)
- [ ] **MLOps 파이프라인** 도입 (MLflow, Docker)
- [ ] **모바일 앱** 개발 (React Native)
- [ ] **지역별 상세 예측** 모델 고도화
- [ ] **AWS/GCP 클라우드** 배포

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 문의

**프로젝트 관련 문의사항이 있으시면 언제든 연락주세요!**

- **GitHub**: [@k-j-hyun](https://github.com/k-j-hyun)
- **Email**: spellrain@naver.com

---

<div align="center">

**이 프로젝트가 도움이 되셨다면 스타를 눌러주세요!**

</div>
