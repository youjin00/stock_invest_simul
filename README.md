# 단기 고성과 팩터 모델 백테스팅

단기에 성과가 좋은 팩터들을 활용하여 스크리닝 후 백테스팅 하였습니다. 팩터 선별의 경우, 삼성증권 2024년 3월 퀀트 투자 보고서를 활용하였습니다. 이후 NAVER 증권의 재무 정보를 웹 스크래핑하는 알고리즘을 만들어 팩터들을 스크리닝하고 백테스팅하였습니다.

---

### **NAVER Stock 웹 스크래핑**

python + html5를 활용하여 웹 스크래핑을 하였습니다. 크롤링 라이브러리로는 requests를 활용하였고, pandas의 html 기능을 활용하여 웹 데이터를 추출하였습니다. 

![stock_invest_simul/naver_financial_con.ipynb at main · youjin00/stock_invest_simul](https://github.com/youjin00/stock_invest_simul/blob/main/naver_financial_con.ipynb)

추출 후 결과물의 일부는 아래와 같습니다. 

컨센서스 자료 역시 받아올 수 있도록 구현하였습니다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/382d9196-6f87-4398-affc-883d4cc1de61/Untitled.png)

### **팩터 선별 및 스크리닝**

투자 전략에 대한 정보는 아래와 같습니다. 

퀀트 투자가 처음이었기 때문에 백테스트 결과만 좋은 종목에 치중했다는 한계점이 존재합니다. 

![14기 윤유진 3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/bde3f2ee-ac2f-49dc-8f39-63e19c61e727/14%E1%84%80%E1%85%B5_%E1%84%8B%E1%85%B2%E1%86%AB%E1%84%8B%E1%85%B2%E1%84%8C%E1%85%B5%E1%86%AB_3.png)

![14기 윤유진 4.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/42d679fa-bd46-429d-b068-7b7ab0ff7af6/14%E1%84%80%E1%85%B5_%E1%84%8B%E1%85%B2%E1%86%AB%E1%84%8B%E1%85%B2%E1%84%8C%E1%85%B5%E1%86%AB_4.png)

![14기 윤유진 6.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/11a8f543-a28b-4494-98ce-12741d21f77d/14%E1%84%80%E1%85%B5_%E1%84%8B%E1%85%B2%E1%86%AB%E1%84%8B%E1%85%B2%E1%84%8C%E1%85%B5%E1%86%AB_6.png)

![stock_invest_simul/final/screening/screening_update.ipynb at main · youjin00/stock_invest_simul](https://github.com/youjin00/stock_invest_simul/blob/main/final/screening/screening_update.ipynb)

실제 스크리닝을 시행할 때는, pykrx 모듈을 활용하였습니다.

### **백테스트**

백테스트 기간은 2010년부터로 잡았습니다. 

다만, 퀀트가 처음이었기에 구현 당시 2008년부터 잡지 않은 것이 아쉽습니다. 

![backtest.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/60c516c3-f4f5-4664-a9b3-c5fdfce32594/7d2ca0bd-b294-4a56-bc9a-d1cd4428217d/backtest.png)

![stock_invest_simul/final_code.ipynb at main · youjin00/stock_invest_simul](https://github.com/youjin00/stock_invest_simul/blob/main/final_code.ipynb)

전체 코드와 데이터는 아래 깃허브 사이트에 백업해 놓았습니다.

![GitHub - youjin00/stock_invest_simul: DART 모의투자 전략 구현 코드](https://github.com/youjin00/stock_invest_simul/tree/main)

---

*Copyright 2023. Yujin Yun All rights reserved.*
