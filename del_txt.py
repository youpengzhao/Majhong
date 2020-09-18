#删除 should_delete.txt 中记录的文件

import os
for line in open("should_delete.txt"):
    num = str(int(line))
    print(int(line))
    os.remove("./p0winresult/("+num+").txt")
    os.remove("./p0winresult_tezheng/("+num+").txt")
print("Finish")