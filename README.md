# Web Retrieval and Mining
## Programming HW1

By: Wu-Jun Pei (B06902029)

### Useful Links
- [Kaggle](https://www.kaggle.com/c/wm-2020-vsm-model)
- [Lecture Website](https://www.csie.ntu.edu.tw/~pjcheng/course/wm2020/)

### Report
Check -> [report](./report.md)

### Execute
- Compile
    ```bash
./compile.sh
```
- Execute
```bash
./execute -i query-file \
          -o output-file \
          -m model-dir \
          -d CTCIR-dir
```

- Example
```bash
./execute.sh -i DATA/queries/query-train.xml \
             -o ./ranked-list-train.csv \
             -m DATA/model \
             -d DATA/CIRB010/
```

