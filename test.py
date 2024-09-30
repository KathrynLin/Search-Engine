mwe_filepath = '/Users/linfangqing/Desktop/SI650/HW/HW1/starter-code/tests/multi_word_expressions.txt'
mwe_list = []
with open(mwe_filepath, 'r') as f: 
    lines = f.readlines()
    for line in lines:
        mwe_list.append(line.strip())
        
mwe_list.sort(key=len, reverse=True)
multiword_expressions_tokens = [token.split() for token in mwe_list]
#print(multiword_expressions_tokens)
for phrase in multiword_expressions_tokens:
    #print(phrase)
    pass

import string
if "'" in string.punctuation:
    print("yes")
else:
    print("no")
    
    
    
import unittest
from document_preprocessor import SplitTokenizer
class TestTokenizer(unittest.TestCase):
    
    def test_MWE_with_punc(self):
        """测试包含标点符号的多词表达式标记化"""
        
        # 定义多词表达式
        multiword_expressions = ["United Nations", "United Nations Children's Fund"]
        
        # 输入带标点符号的文本
        input_tokens = ['UNICEF', ',', 'now', 'officially', 'United', 'Nations', 'Children', "'s", 'Fund', ',', 'is', 'a', 'United', 'Nations', 'agency', '.']
        
        # 预期输出
        expected_tokens = ['UNICEF', ',', 'now', 'officially', "United Nations Children's Fund", ',', 'is', 'a', 'United Nations', 'agency', '.']
        
        # 初始化带有多词表达式的分词器
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=multiword_expressions)

        # 调用 postprocess 处理输入
        processed_tokens = tokenizer.postprocess(input_tokens)
        
        # 断言处理后的结果是否与预期结果相符
        self.assertEqual(processed_tokens, expected_tokens)

# 运行测试用例
if __name__ == '__main__':
    unittest.main()