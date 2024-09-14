# Long Term Writing

전체적인 activity diagram은 아래와 같습니다. 여기에서는 plan and excute 패턴을 가지는 agent와 reflection을 수행하는 agent를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. Multi agent 구조로 구성함으로써 복잡한 workflow를 단순하게 구현할 수 있습니다.

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6fe65b1b-a591-4eae-af28-4b5d028774c5">

사용자의 instruction은 plan_node에서 n개의 plan을 생성합니다. execution_node는 instruction, plans와 현재의 step을 이용하여 draft를 생성합니다. n개의 draft들이 생성됩니다.

<img width="200" alt="image" src="https://github.com/user-attachments/assets/2020f67e-53bd-4d10-995d-d88c952f7f83">

revise_node에서는 drafts를 각각 reflect_node에서 reflections을 추출합니다. 또한 이때 최대 3개의 search_queries도 함께 추출하여 검색을 통해 contents를 수집합니다. reflection과 search_queries에 대한 contents를 이용하여 revise_answer에서는 질문을 업데이트합니다. 

![image](https://github.com/user-attachments/assets/be4efa7d-8e93-419e-a46c-2c0eb9f41400)


## 실행결과

300초 소요.

![image](https://github.com/user-attachments/assets/59f105f1-4777-471f-8416-5826c703f254)

"AWS의 Cloud를 이용해 ERP를 구축하는 방법에 대해 정리해줘."라고 입력합니다.

![image](https://github.com/user-attachments/assets/5ca3a6cd-fced-418d-b9e2-31842a3d6662)


