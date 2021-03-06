## Docker Commands

#### 1.列出所有執行中的container
```
docker ps
```

---
#### 2.列出被使用過名稱的container
```
docker ps -a
```
---
#### 3.停止特定container
* 先找出現在有哪些container, 輸入
```
`docker stats`
```
* 會顯示類似如下
```
CONTAINER           CPU %               MEM USAGE / LIMIT       MEM %               NET I/O             BLOCK I/O           PIDS
01f51f8c9f7b        30.80%              326.6 MiB / 1.952 GiB   16.34%              3.6 MB / 522 kB     9.84 MB / 0 B       12
```
* 接著輸入要stop的container
```
docker stop 01f51f8c9f7b
```
---
#### 4.停止所有container
```
docker stop (docker ps -a -q)
```
---
#### 5.移除所有使用名稱的container
在kill或stop container後要再把名稱移除才可再次重新使用
```
docker rm (docker ps -q -f status=exited)
```
---
#### 6.以名字顯示running中的container
```
docker stats $(docker ps --format={{.Names}})
```
