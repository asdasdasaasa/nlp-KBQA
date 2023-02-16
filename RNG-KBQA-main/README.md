# RNG-KBQA
Generation Augmented Iterative Ranking for Knowledge Base Question Answering
Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base  Embeddings (Slides)

场景：通过在知识图谱上查询知识来回答自然语言问题
背景：常用的Multi-hop KGQA方法往往限制了候选答案的跳数范围，比如说在问题中实体的3-hop范围内，
这种方法会因为正确答案不在范围内而无法得到正确答案。在稀疏的、不完整的KG中,这种情况时常出现。
在知识图谱领域,常利用知识图谱嵌入来进行链接预测工作，减轻KG的稀疏度。本文想通过利用知识图谱嵌入来克服目前
Multi-hop KGQA的这些限制，提出EmbedKGQA。
