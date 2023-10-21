
'''
transformers导入模型，相对路径则是的从网址上下载模型 绝对路径则是本地导入
项目中所有文件夹命名不能出现中文，否则会报错
'''



'''
模型结构改变。导入后加入新的层
'''
import torch
from transformers import AutoTokenizer, AutoModel
class  Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(r"D:\pythonProject\test\chatglm\THUDM\chatglm2-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(r"D:\pythonProject\test\chatglm\THUDM\chatglm2-6b", trust_remote_code=True).half().cuda()
        self.fc =torch.nn.Linear(in_features=768,out_features=32)
    def forward(self):
        return 1

m= Model()
print(m)
'''
修改模型某个层的方法的
'''
import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(r"D:\pythonProject\test\chatglm\THUDM\chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(r"D:\pythonProject\test\chatglm\THUDM\chatglm2-6b", trust_remote_code=True).half().cuda()
model.pooler.dense = torch.nn.Linear(in_features=768,out_features=32)
print(model)

'''
修改模型某个层的方法的
'''
import torch.nn
from pytorch_pretrained_bert import BertModel
model = BertModel.from_pretrained(r"D:\pythonProject\test\chatglm\THUDM\chatglm2-6b")
print(model)
print('~~'*100)
# for para in model.parameters():
#     para.requires_grad=True

# model.pooler.dense = torch.nn.Linear(in_features=768,out_features=32)
# print(model)


from torchsummary import summary

summary(model)


