# State Diagram

[state-diagram.plantuml](https://github.com/joelparkerhenderson/plantuml-examples/blob/master/doc/state-diagram/state-diagram.plantuml.txt)

```text
@startuml
skinparam monochrome true
[*] --> State1 : Start
State1 -> State2 : Change1
State2 -> State3 : Change2
State3 --> [*] : Stop
State1 : Description 1
State2 : Description 2
State3 : Description 3
@enduml
```

이때의 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/be9ea374-6641-44a6-98f9-db569f2ec411)


[activity-diagram.plantuml](https://github.com/joelparkerhenderson/plantuml-examples/blob/master/doc/activity-diagram/activity-diagram.plantuml.txt)

```text
# State Diagram

@startuml
skinparam monochrome true
start
-> Starting;
:Activity 1;
if (Question) then (yes)
  :Option 1;
else (no)
  :Option 2;
endif
:Activity 2;
-> Stopping;
stop
@enduml
```

이때의 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/f6832e19-315b-4cec-8631-9885a003fecd)

### Activity Diagram

[activity-diagram](https://github.com/mattjhayes/PlantUML-Examples/blob/master/docs/Misc/BlogSource/activity-diagram.txt)

![image](https://github.com/user-attachments/assets/b7386a41-40f2-4a54-9285-553c72922158)

```text
@startuml
skinparam shadowing false

title Activity Diagram Example\n

skinparam activity {
    StartColor limegreen
    EndColor darkblue
    BackgroundColor #d4de5e
    BorderColor #5e94de
    ArrowColor black
}
skinparam activityDiamond {
    BackgroundColor #5ede68
    BorderColor #5e94de
    fontSize 16
}

start
:choose diagram type to suit 
message and audience;
:choose example diagram
and copy code;

while (Is diagram finished?) is (not finished)
    :update diagram code;
    :render and review diagram;
endwhile (finished)

:publish diagram;

stop

@enduml
```

## C4-PlantUML

[C4-PlantUML](https://github.com/plantuml-stdlib/C4-PlantUML)을 참조합니다.

![image](https://github.com/user-attachments/assets/f77820fb-9391-4af1-bfef-7b24e91bc791)


```text
@startuml Self RAG

!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

skinparam wrapWidth 200
skinparam maxMessageSize 200

!$BLACK = "#black"
!$COLOR_A_6 = "#d4de5e"
!$COLOR_A_5 = "#7f3b08"
!$COLOR_A_4 = "#b35806"
!$COLOR_A_3 = "#e08214"
!$COLOR_A_2 = "#fdb863"
!$COLOR_A_1 = "#fee0b6"
!$COLOR_NEUTRAL = "#f7f7f7"
!$COLOR_B_1 = "#d8daeb"
!$COLOR_B_2 = "#b2abd2"
!$COLOR_B_3 = "#8073ac"
!$COLOR_B_4 = "#542788"
!$COLOR_B_5 = "#2d004b"
!$COLOR_REL_LINE = "#8073ac"
!$COLOR_REL_TEXT = "#8073ac"

UpdateElementStyle("container", $bgColor=$COLOR_A_6, $fontColor=$BLACK, $borderColor=$COLOR_A_1, $shadowing="false", $legendText="Internal user")
UpdateElementStyle("system", $bgColor=$COLOR_A_4, $fontColor=$COLOR_NEUTRAL, $borderColor=$COLOR_A_2, $sprite="robot", $legendText="Our chatbot based system")
UpdateElementStyle("system", $bgColor=$COLOR_B_4, $fontColor=$COLOR_NEUTRAL, $borderColor=$COLOR_B_2, $legendText="External system")
UpdateRelStyle($lineColor=$COLOR_REL_LINE, $textColor=$COLOR_REL_TEXT)

LAYOUT_WITH_LEGEND()

Container(start, "Start")


'System_Boundary("start", ""){
    Container(retrieve, "retrieve")
    
    Container(grade_documents, "grade documents")

    SystemQueue_Ext(decide_to_generate, "decide_to_generate")

    Container(generate, "generate")

    SystemQueue_Ext(grade_generation, "grade generation")
'}

Container(rewrite, "rewrite")


Rel(start, retrieve, "question")

Rel(retrieve, grade_documents, "documents")

Rel(grade_documents, decide_to_generate, "grade")

Rel(decide_to_generate, rewrite, "no")

Rel(decide_to_generate, generate, "yes")

Rel(rewrite, retrieve, "update \nquestion")

Rel(generate, grade_generation, "answer")

Rel(grade_generation, generate, "not support")

Rel(grade_generation, END, "useful")

Rel(grade_generation, rewrite, "not useful")


'Container(websearch, "web search")




' fight the layout engine
'auth -[hidden]right-> shop


@enduml
```
