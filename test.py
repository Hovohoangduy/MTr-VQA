import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import matplotlib.pyplot as plt
from utils.config import Config
from model.mtr_model import ImageEmbedding, QuesEmbedding, AnsEmbedding
from model.sans import StackAttention
from model.decoder import Decoder
from model.vqa_model import VQAModel
from utils.data_processing import test_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_rouge(references, hypotheses, scorer):
    total_rouge_l = 0.0
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(' '.join(hyp), ' '.join(ref))
        total_rouge_l += scores['rougeL'].fmeasure
    return total_rouge_l / len(references)

def evaluation(model, test_loader, criterion, vocab_swap, device):
    model.eval() 
    total_loss = 0.0
    total_rouge = 0.0
    total_bleu_1 = 0.0
    total_bleu_2 = 0.0
    total_bleu_3 = 0.0
    total_bleu_4 = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            anno_id, img_id, images, questions, answers = batch
            if len(images) == Config.TRAIN_BATCH_SIZE:
                predicted_tokens, ans_embedds = model(images.to(device), questions, answers, anno_id, mode='train', mask=True)
                predicted_tokens = predicted_tokens.float()
                ans_embedds = ans_embedds.long()

                # Compute ROUGE and BLEU
                references = [answer.split() for answer in answers]
                hypotheses = []
                for i in range(Config.TRAIN_BATCH_SIZE):
                    sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                    predicted_sentence = ""
                    for idx in sentence_predicted:
                        predicted_sentence += vocab_swap[idx.item()] + " "
                        if idx == 2:
                            break
                            
                    predicted_sentence = predicted_sentence.strip()
                    hypotheses.append(predicted_sentence.split())
                rouge_score = compute_rouge(references, hypotheses, scorer)
                bleu_score_1 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
                bleu_score_2 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
                bleu_score_3 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother.method1)
                bleu_score_4 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1)
                total_rouge += rouge_score
                total_bleu_1 += bleu_score_1
                total_bleu_2 += bleu_score_2
                total_bleu_3 += bleu_score_3
                total_bleu_4 += bleu_score_4

                total_loss += criterion(predicted_tokens.permute(0, 2, 1), ans_embedds).item()

    avg_loss = total_loss / len(test_loader)
    avg_rouge = total_rouge / len(test_loader)
    avg_bleu_1 = total_bleu_1 / len(test_loader)
    avg_bleu_2 = total_bleu_2 / len(test_loader)
    avg_bleu_3 = total_bleu_3 / len(test_loader)
    avg_bleu_4 = total_bleu_4 / len(test_loader)

    return avg_loss, avg_rouge, avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
    vocab = tokenizer.get_vocab()

    model = VQAModel().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    vocab_swap = {value: key for key, value in vocab.items()}

    test_loss, test_rouge, test_bleu_1, test_bleu_2, test_bleu_3, test_bleu_4 = evaluation(model, test_loader, criterion, vocab_swap, device)

    print(f"Test ROUGE: {test_rouge:.4f}")
    print(f"Test BLEU@1: {test_bleu_1:.4f}")
    print(f"Test BLEU@2: {test_bleu_2:.4f}")
    print(f"Test BLEU@3: {test_bleu_3:.4f}")
    print(f"Test BLEU@4: {test_bleu_4:.4f}")