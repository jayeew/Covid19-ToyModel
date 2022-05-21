# Covid19-ToyModel
A toy model used to predict the 'confirmedIncr' based on China-Covid19-province data.

## Requirement
This project should be able to run without any modification after following packages installed.  
```
pytorch
numpy
```

## Datasets supported
China-Covid19-province data from ["Code-For-COVID-19-Data"](https://github.com/eAzure/Code-For-COVID-19-Data). For example,
```
北京市.json
```

## Field description：
- id：数据编号
- confirmedCount：累计确诊
- confirmedIncr：新增确诊
- curedCount：累计治愈
- curedIncr：新增治愈
- currentConfirmedCount：现存确诊
- currentConfirmedIncr：新增现存确诊
- dateId：日期
- deadCount：累计死亡
- deadIncr：新增死亡
- suspectedCount：累计疑似
- suspectedCountIncr：新增疑似

## Visualization
![](https://github.com/jayeew/Covid19-ToyModel/blob/main/pic/beijing.png)
