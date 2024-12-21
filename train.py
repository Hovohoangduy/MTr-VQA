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
from utils.data_processing import train_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_rouge(references, hypotheses, scorer):
    total_rouge_l = 0.0
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(' '.join(hyp), ' '.join(ref))
        total_rouge_l += scores['rougeL'].fmeasure
    return total_rouge_l / len(references)

def train_model(model, train_loader, criterion, optimizer, scheduler, config, vocab_swap, device):
    # Initialize SmoothingFunction
    smoother = SmoothingFunction()

    num_epochs = config.EPOCHS
    print_every = 500

    losses = []
    rouge_scores = []  
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    # Initialize scorer for ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0.0
        total_rouge = 0.0
        total_bleu_1 = 0.0
        total_bleu_2 = 0.0
        total_bleu_3 = 0.0
        total_bleu_4 = 0.0

        for batch_idx, batch in enumerate(train_loader):
            anno_id, img_id, images, questions, answers = batch
            if len(images) == config.TRAIN_BATCH_SIZE:
                predicted_tokens, ans_embedds = model(images.to(device), questions, answers, anno_id, mode='train', mask=True)
                predicted_tokens = predicted_tokens.float()
                ans_embedds = ans_embedds.long()

                # Compute ROUGE and BLEU
                references = [answer.split() for answer in answers]
                hypotheses = []
                for i in range(config.TRAIN_BATCH_SIZE):
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

                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    print(f"ROUGE: {rouge_score:.4f}")
                    print(f"BLEU@1: {bleu_score_1:.4f}")
                    print(f"BLEU@2: {bleu_score_2:.4f}")
                    print(f"BLEU@3: {bleu_score_3:.4f}")
                    print(f"BLEU@4: {bleu_score_4:.4f}")
                    for i in range(config.TRAIN_BATCH_SIZE):
                        sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                        predicted_sentence = ""
                        for idx in sentence_predicted:
                            predicted_sentence += vocab_swap[idx.item()] + " "
                            if idx == 2:
                                break

                        predicted_sentence = predicted_sentence.strip()
                        print(f"Question: {questions[i]}")
                        print(f"Answer: {answers[i]}")
                        print(f"Answer Prediction: {predicted_sentence}")
                    print("\n")

                # Compute loss and update model
                loss = criterion(predicted_tokens.permute(0, 2, 1), ans_embedds)
                valid_indicies = torch.where(ans_embedds == 1, False, True)
                loss = loss.sum() / valid_indicies.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                losses.append(loss.item())

        avg_rouge = total_rouge / len(train_loader)
        avg_bleu_1 = total_bleu_1 / len(train_loader)
        avg_bleu_2 = total_bleu_2 / len(train_loader)
        avg_bleu_3 = total_bleu_3 / len(train_loader)
        avg_bleu_4 = total_bleu_4 / len(train_loader)

        rouge_scores.append(avg_rouge)
        bleu1_scores.append(avg_bleu_1)
        bleu2_scores.append(avg_bleu_2)
        bleu3_scores.append(avg_bleu_3)
        bleu4_scores.append(avg_bleu_4)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Average ROUGE: {avg_rouge:.4f}")
        print(f"Average BLEU@1: {avg_bleu_1:.4f}")
        print(f"Average BLEU@2: {avg_bleu_2:.4f}")
        print(f"Average BLEU@3: {avg_bleu_3:.4f}")
        print(f"Average BLEU@4: {avg_bleu_4:.4f}")

        # plot training for epochs
        plt.figure(figsize=(10, 6))
        plt.plot(rouge_scores, label='ROUGE', marker='o')
        plt.plot(bleu1_scores, label='BLEU@1', marker='o')
        plt.plot(bleu2_scores, label='BLEU@2', marker='o')
        plt.plot(bleu3_scores, label='BLEU@3', marker='o')
        plt.plot(bleu4_scores, label='BLEU@4', marker='o')
        plt.title('Evaluation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('evaluation_metrics_plot.png')
        plt.show()

        # plot loss
        plt.figure(figsize=(10, 4))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.xlabel('Batch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training Loss Over Batches', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
    vocab = tokenizer.get_vocab()

    model = VQAModel().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=int(len(train_loader) * Config.EPOCHS)
    )

    vocab_swap = {value: key for key, value in vocab.items()}

    train_model(model, train_loader, criterion, optimizer, scheduler, Config, vocab_swap, device)