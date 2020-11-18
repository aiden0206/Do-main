이 활동은 스타트업 domain에서 제작한 하드웨어로부터 수집된 EOG(electrooculography)에서 안구 운동 상태를 추정하는 것입니다.  
EOG_HMM.py: 주 알고리즘으로 은닉 마르코프 모델을 사용하였습니다. https://github.com/jason2506/PythonHMM의 hmm.py 코드를 기반으로 했으나 활동 용도에 맞게 혼합 가우시안 방출 분포를 갖도록 확장했습니다.   
preprocess.py: HMM 모델의 성능을 높이기 위해 제작한 뇌파 전처리 코드입니다. 기본적인 정규화와 함께 성능 저하의 주 원인인 베이스라인 드리프트를 수정하는 메소드가 있습니다.  
test.py: 성능 테스트에 사용된 코드입니다. 아래 사진과 같은 결과를 얻습니다. m은 해당 시점에서 안구가 중앙에 위치함, u는 위, d는 아래에 위치함을 뜻합니다.  
train.py 모델 학습에 사용된 코드입니다.

<img width="981" alt="result" src="https://user-images.githubusercontent.com/62476546/99537335-e112e800-29ee-11eb-93ea-dd6592cbfc8c.png">