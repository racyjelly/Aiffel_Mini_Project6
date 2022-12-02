import os, re
import numpy as np
import tensorflow as tf

file_path = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject6\data\shakespeare.txt"
with open(file_path, "r") as f:
    raw_corpus=f.read().splitlines()

for idx, sentence in enumerate(raw_corpus):
    if len(sentence)==0: continue
    if sentence[-1]==":":continue # 문장 끝이 :인 문장은 건너뜀

    if idx>9: break
    print(sentence)
print('End to search')

# 입력된 문장을
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip() 
    # 1. 소문자로 바꾸고, 양쪽 공백을 지웁니다
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    # 2. 특수문자 양쪽에 공백을 넣고
    sentence = re.sub(r'[" "]+', " ", sentence) 
    # 3. 여러개의 공백은 하나의 공백으로 바꿉니다
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence) 
    # 4. a-zA-Z?.!,¿가 아닌 모든 문자를 하나의 공백으로 바꿉니다
    sentence = sentence.strip() 
    # 5. 다시 양쪽 공백을 지웁니다
    sentence = '<start> ' + sentence + ' <end>'
    # 6. 문장 시작에는 <start>, 끝에는 <end>를 추가합니다
    return sentence
    # 이 순서로 처리해주면 문제가 되는 상황을 방지할 수 있음

# 이 문장이 어떻게 필터링되는지 확인해 보세요.
print(preprocess_sentence("This @_is ;;;sample        sentence."))

# 정제 문장 모으기
corpus = []
# 정제안된 코퍼스 리스트에 저장된 문장들을 순서대로 반환하여 setence에 저장
for sentence in raw_corpus:
    if len(sentence)==0: continue
    if sentence[-1]==":": continue
    # 앞서 구현한 정제 함수를 이용하여 문장 정제하고 담기
    pre_sentence = preprocess_sentence(sentence)
    corpus.append(pre_sentence)

print(corpus[:10])

# 토큰화할때 텐서플로우의 Tokenizer와 pad_sequences를 사용
def tokenize(corpus):
    # 7000단어를 기억할 수 있는 tokenizer를 만들기
    # 우리는 이미 문장을 정제했으니 filters가 필요없음!
    # 7000단어에 포함되지 못한 단어는 '<unk>'로 바꾸기!
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=7000, 
        filters=' ',
        oov_token="<unk>"
    )
    # corpus를 이용해 tokenizer 내부의 단어장을 완성합니다
    # tokenizer.fit_on_texts(texts): 문자 데이터를 입력받아 리스트의 형태로 변환하는 메서드
    tokenizer.fit_on_texts(corpus)
    # 준비한 tokenizer를 이용해 corpus를 Tensor로 변환합니다
    # tokenizer.texts_to_sequences(texts): 텍스트 안의 단어들을 숫자의 시퀀스 형태로 변환하는 메서드
    tensor = tokenizer.texts_to_sequences(corpus)   
    # 입력 데이터의 시퀀스 길이를 일정하게 맞춰줍니다
    # 만약 시퀀스가 짧다면 문장 뒤에 패딩을 붙여 길이를 맞춰줍니다.
    # 문장 앞에 패딩을 붙여 길이를 맞추고 싶다면 padding='pre'를 사용합니다
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  
    
    print(tensor,tokenizer)
    return tensor, tokenizer

tensor, tokenizer = tokenize(corpus)
# tensor data는 tokenizer에 구축된 단어 사전의 인덱스

print(tensor[:3, :10])

for idx in tokenizer.index_word:
    print(idx, ":", tokenizer.index_word[idx])

    if idx >=10: break

print(len(tokenizer.index_word))

# 텐서 출력부에서 행 뒤쪽에 0이 많이 나온 부분은 정해진 입력 시퀀스 길이보다 문장이 짧을 경우 0으로 패딩(padding)을 채워 넣은 것
# 사전에는 없지만 0은 바로 패딩 문자 <pad>가 될 것입니다.
# tensor에서 마지막 토큰을 잘라내서 소스 문장을 생성합니다
# 마지막 토큰은 <end>가 아니라 <pad>일 가능성이 높습니다.
src_input = tensor[:, :-1]  
# tensor에서 <start>를 잘라내서 타겟 문장을 생성합니다.
tgt_input = tensor[:, 1:]    

print(src_input[0])
print(tgt_input[0])

buffer_size = len(src_input)
batch_size = 256
steps_per_epoch = len(src_input)//batch_size

vocab_size = tokenizer.num_words+1

dataset = tf.data.Dataset.from_tensor_slices((src_input, tgt_input))
dataset = dataset.shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset
print(len(dataset))
# 정규표현식을 이용한 corpus 생성
# tf.keras.preprocessing.text.Tokenizer를 이용해 corpus를 텐서로 변환
# tf.data.Dataset.from_tensor_slices()를 이용해 corpus 텐서를 tf.data.Dataset객체로 변환


class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(TextGenerator, self).__init__()
        # Embedding layer, 2개 LSTM layer, 1개 Dense layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn3 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        out = self.embedding(x)
        out = self.rnn1(out)
        out = self.rnn2(out)
        out = self.rnn3(out)
        out = self.linear(out)
        return out

embedding_size = 256
hidden_size = 1024
model = TextGenerator(vocab_size, embedding_size, hidden_size)

for src_sample, tgt_sample in dataset.take(1): break

model(src_sample)
model.summary() # 32millio정도


# 손실함수의 최소값을 찾는 것을 학습의 목표로 하며 여기서 최소값을 찾아가는 과정을 optimization
# 이를 수행하는 알고리즘을 optimizer(최적화)라고 함


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
loss = tf.keras.losses.SparseCategoricalCrossentropy( # 훈련 데이터의 라벨이 정수의 형태로 제공될 때 사용하는 손실함수이다.
    from_logits=True, # 기본값은 False이다. 모델에 의해 생성된 출력 값이 정규화되지 않았음을 손실 함수에 알려줌. 
    # 즉 softmax함수가 적용되지 않았다는걸 의미
    reduction='none'  # 기본값은 SUM이다. 각자 나오는 값의 반환 원할 때 None을 사용한다.
)
# 모델 학습
model.compile(loss=loss, optimizer=optimizer) # 손실함수와 훈련과정을 설정
model.fit(dataset, epochs=30)

def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    test_input = tokenizer.text_to_sequences([init_sentence])
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    # 단어 하나씩 예측해서 문장 만들기
    while True:
        # 1. 입력받은 문장의 텐서를 입력합니다
        predict = model(test_tensor) 
        # 2. 예측된 값 중 가장 높은 확률인 word index를 뽑아냅니다
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1]
        # 3. 2에서 예측된 word index를 문장 뒤에 붙입니다
        test_tensor = tf.cocncat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)
        # 4. 모델이 <end>를 예측했거나, max_len에 도달했다면 문장 생성을 마칩니다
        # (도달 하지 못하였으면 while 루프를 돌면서 다음 단어를 예측)
        if predict_word.numpy()[0]==end_token: break
        if test_tensor.shape[1] >= max_len: break
    
    generated = ""
    # tokenizer를 이용해 word index를 단어로 하나씩 변환
    for word_index in test_tensor[0].numpy():
        generated += tokenizer.index_word[word_index] + " "
    
    return generated # 최종적으로 모델이 생성한 문장 반환

print(generate_text(model, tokenizer, init_sentence="<start> he"))
# 시작문장으로 he를 넣어 문장생성 함수 실행
print(generate_text(model, tokenizer, init_sentence="<start> she"))
