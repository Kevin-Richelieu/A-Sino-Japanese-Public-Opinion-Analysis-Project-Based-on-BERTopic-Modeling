import re
import os
from transformers import pipeline
import xlsxwriter

DATASET_DIR = r'' 
MODEL_PATH = r'' # bert-base-japanese-v3-finetuned-sentiment
EXCEL_NAME = 'SentimentAnalysis.xlsx'

nlp = pipeline("sentiment-analysis", model=MODEL_PATH)
workbook = xlsxwriter.Workbook(EXCEL_NAME)
worksheet = workbook.add_worksheet()

worksheet.write(0, 0, '日语句子')
worksheet.write(0, 1, '情感极性')
worksheet.write(0, 2, '极性信度')
current_row = 1

for filename in os.listdir(DATASET_DIR):
    file_full_path = os.path.join(DATASET_DIR, filename)
    if os.path.isfile(file_full_path):
        try:
            with open(file_full_path, 'r', encoding='utf-8') as f:
                textdata = f.read()
        except Exception as e:
            continue

        sentencelist = re.split('[。?!.\n]', textdata)
        for s in sentencelist:
            s = s.strip()
            if s and len(s) > 1:
                result = nlp(s)
                label = result[0]['label']
                score = round(result[0]['score'], 4)

                worksheet.write(current_row, 0, s)
                if label == 'LABEL_0':
                    worksheet.write(current_row, 1, '十分消极')
                elif label == 'LABEL_1':
                    worksheet.write(current_row, 1, '比较消极')
                elif label == 'LABEL_2':
                    worksheet.write(current_row, 1, '中性')
                elif label == 'LABEL_3':
                    worksheet.write(current_row, 1, '比较积极')
                elif label == 'LABEL_4':
                    worksheet.write(current_row, 1, '十分积极')
                worksheet.write(current_row, 2, score)
                current_row += 1

workbook.close()