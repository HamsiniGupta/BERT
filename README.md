# BERT

Large Language Models struggle with biomedical Q&A due to hallucinations and outdated knowledge. This research addresses these limitations by developing PubMedRAG, a domain-specific retriever model trained using Simple Contrastive Sentence Embeddings (SimCSE) on the PubMedQA labeled dataset. We compare this to its baseline BERT model to evaluate the effects of SimCSE.

## Key Results

- **53% accuracy** with Llama3-OpenBioLLM-8B
- **38% accuracy** with Llama-3.1-8B  
- Baseline results

## Installation

```bash
git clone https://github.com/HamsiniGupta/BERT
pip install -r requirements.txt
```

## Evaluation
```bash
# Get results for BERT
cd BERT_Files/BERT_test
python testBERT.py
# Run PubMedRAG files and compare results with BERT
python compareAllEmbeddings.py
```
The following table shows the results for different retrievers in the RAG pipeline evaluated on both LLMs.
<img width="767" height="385" alt="image" src="https://github.com/user-attachments/assets/8d0ad4ad-dc5f-44f3-9699-8678f5203c20" />


## Acknowledgements
This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.
