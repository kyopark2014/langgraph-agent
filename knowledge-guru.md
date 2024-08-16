# Knoeledge Guru

Knowledge Guru의 activity diagram은 아래와 같습니다. 

1) 요청(task)에 따라 tools를 이용하여 draft 답변을 구합니다.
2) draft로 부터 검색에 사용할 keyword를 추출하고, 각 keyword로 search를 포함한 tools에 요청하여 관련된 정보를 얻습니다. 
3) 관련된 정보(content)를 이용하여 답변을 업데이트(revise) 합니다.
4) 2번 3번의 과정을 max_revision 만큼 반복합니다.

![image](https://github.com/user-attachments/assets/009a20ec-0993-4be3-a8ac-a5f9b7f04a68)
