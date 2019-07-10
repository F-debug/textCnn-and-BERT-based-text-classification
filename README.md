# textCnn-and-BERT-based-text-classification
TextCNN模型是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛，我在项目组工作中，使用TextCNN对文本分类取得了不错的准确率。BERT模型是我最近研究的一个分类模型。BERT拥有一个深而窄的神经网络。transformer的中间层有2018，BERT只有1024，但却有12层。因此，它可以在无需大幅架构修改的前提下进行双向训练。由于是无监督学习，因此不需要人工干预和标注，让低成本地训练超大规模语料成为可能。

#textcnn Introduce
TextCnn在文本分类问题上有着更加卓越的表现。从直观上理解，TextCNN通过一维卷积来获取句子中n-gram的特征表示。TextCNN对文本浅层特征的抽取能力很强，在短文本领域如搜索、对话领域专注于意图分类时效果很好，应用广泛，且速度快，一般是首选；对长文本领域，TextCNN主要靠filter窗口抽取特征，在长距离建模方面能力受限，且对语序不敏感。  
在项目中，我们发现TextCNN是一个n-gram特征提取器，对于训练集中没有的n-gram不能很好的提取。对于有些n-gram，可能过于强烈，反而会干扰模型，造成误分类。TextCNN对词语的顺序不敏感，在query推荐中，我把正样本分词后得到的term做随机排序，正确率并没有降低太多，当然，其中一方面的原因短query本身对term的顺序要求不敏感。隔壁组有用textcnn做博彩网页识别，正确率接近95%，在对网页内容（长文本）做随机排序后，正确率大概是85%。TextCNN擅长长本文分类，在这一方面可以做到很高正确率。TextCNN在模型结构方面有很多参数可调。  

#BERT Introduce
BERT拥有一个深而窄的神经网络。transformer的中间层有2018，BERT只有1024，但却有12层。因此，它可以在无需大幅架构修改的前提下进行双向训练。由于是无监督学习，因此不需要人工干预和标注，让低成本地训练超大规模语料成为可能。  
BERT模型能够联合神经网络所有层中的上下文来进行训练。这样训练出来的模型在处理问答或语言推理任务时，能够结合上下文理解语义，并且实现更精准的文本预测生成。  
BERT只需要微调就可以适应很多类型的NLP任务，这使其应用场景扩大，并且降低了企业的训练成本。BERT支持包括中文在内的60种语言，研究人员也不需要从头开始训练自己的模型，只需要利用BERT针对特定任务进行修改，在单个云TPU上运行几小时甚至几十分钟，就能获得不错的分数。  
虽然BERT模型看起来很美好，但它需要的代价也非常巨大  


#Pre-classification preparation
 数据预处理:data_process.py.去除停用词，标点符号等。生成符合word2vec模型的输入数据  
 预训练word2vec词向量:gen_w2v.py.将训练出的词向量保存为word2Vec.bin
 #textcnn模型  
 textcnn模型代码textCnn.py:参数配置，数据预处理，训练模型  
 
 
 #BERT模型
 BERT模型是基于双向Transformer实现的语言模型，集预训练和下游任务于一个模型中， 因此在使用的时候我们不需要搭建自己的下游任务模型，直接用BERT模型即可，我们将谷歌开源的源码下载 下来放在bert文件夹中，在进行文本分类只需要修改run_classifier.py文件即可，另外我们需要将训练集 和验证集分割后保存在两个不同的文件中，放置在/BERT/data下。然后还需要下载谷歌预训练好的模型放置在 /BERT/modelParams文件夹下，还需要建一个/BERT/output文件夹用来放置训练后的模型文件
 做完上面的步骤之后只要执行下面的脚本即可

  export BERT_BASE_DIR=../modelParams/uncased_L-12_H-768_A-12

  export DATASET=../data/

  python run_classifier.py 
    --data_dir=$MY_DATASET 
    --task_name=imdb 
    --vocab_file=$BERT_BASE_DIR/vocab.txt 
    --bert_config_file=$BERT_BASE_DIR/bert_config.json 
    --output_dir=../output/ 
    --do_train=true 
    --do_eval=true 
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt 
    --max_seq_length=200 
    --train_batch_size=16 
    --learning_rate=5e-5
    --num_train_epochs=3.0
 
