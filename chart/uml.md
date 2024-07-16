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
