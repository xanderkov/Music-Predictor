# Baseline

## Feature extractor

Использовали на спектограмме ResNet50. 

Затем применили LogisticRegression.

Выбраны метрики: macro avg, weighted avg их f1-score - 0.19, 0.12 соотвественно.

Для 21 класса, с multilabel классификации сойдет.

## SIFT

Использовали на спектограмме SIFT. 

Затем применили LogisticRegression.

Выбраны метрики: macro avg, weighted avg

## Анилиз текстов песен

6 жанров. Много текстов.
