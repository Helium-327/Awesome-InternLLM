# Python字符串实战

## 任务一： 处理字符串

- 去除标点符号
- 转小写
- 切分单词
- 去重
- 创建字典输出

```python
# -*- coding: utf-8 -*- 

import re

def wordcount(txt):
    word_dict = {}
    pattern = r"[\.\!\?\n\,\;\:\n]" # 去除标点符号
    text_temp = re.sub(pattern, '', txt)
    
    
    text_temp = text_temp.lower()  # 转小写
    word_list = text_temp.strip().split() # 使用空格切分单词
    # word_list = [w for w in word_list if w != ""] # 去除空字符串
    
    word_set = set(word_list) # 可去除重复，但集合无序
    
    for _ , word in enumerate(word_set):
        k, v = word, word_list.count(word) # 统计每个单词出现的次数
        word_dict[k] = v
    print(word_dict)
```

![image-20240718171843772](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181718815.png)

输出结果为：

```
{'it': 4, 'though': 1, 'day': 1, 'thesame': 1, 'look': 1, 'face': 1, 'friendly': 1, 'play': 1, 'before': 1, 'andsuper': 1, 'earlier': 1, 'itto': 1, 'other': 1, 'everywhere': 1, 'loves': 1, 'takes': 1, 'with': 1, 'theremight': 1, 'to': 1, 'are': 1, 'and': 2, 'bigger': 1, 'got': 2, "daughter's": 1, 'i': 4, 'soft': 1, 'its': 1, 'for': 3, 'paid': 1, 'that': 1, 'think': 1, 'cute': 1, "it'sa": 1, 'price': 1, 'toy': 1, 'expectedso': 1, 'a': 2, 'gave': 1, 'than': 1, 'panda': 1, 'bit': 1, 'small': 1, 'my': 1, 'be': 1, 'this': 1, 'has': 1, 'options': 1, 'plush': 1, 'her': 1, 'what': 1, 'birthdaywho': 1, 'arrived': 1, "it's": 1, 'myself': 1}
```

## 任务二 调试案例

![python](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181828235.gif)